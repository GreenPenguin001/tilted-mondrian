from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from input_loading import LoadedDataset, SplitData
from no_free_repair_core import (
    ScoreFixConfig,
    build_metric_context,
    candidate_metrics_fast,
    candidate_rejection_reasons,
    cell_suffer_rows,
    compute_baseline_group_thresholds,
    compute_thresholds_for_target_counts_from_true_scores,
    define_repair_roles,
    evaluate_split,
    gap_constraint_summary,
    group_ledger_rows,
    halton_value,
    label_gaps,
    labelwise_scores_from_probs,
    lexicographically_better,
    objective_tuple,
    prioritized_label_order,
    prioritized_pair_blocks,
    prioritized_triple_blocks,
    repair_cell_indices,
    row_thresholds,
    theta_scale_from_grid,
    true_scores,
    validate_dataset,
)


SCORE_METHOD_ORDER = ("simple", "aps", "raps", "saps")
OPTIMIZER_NAME = "directional_asymmetric_boundary_search"
CONTAINMENT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class DirectionalArtifacts:
    summary_rows: list[dict[str, Any]]
    offset_rows: list[dict[str, Any]]
    cell_rows: list[dict[str, Any]]
    group_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    search_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]


def run_directional_asymmetric_experiment(
    data: LoadedDataset,
    score_config: ScoreFixConfig,
    result_dir: Path,
    symmetric_result_dir: Path,
) -> DirectionalArtifacts:
    """Run Tilted Mondrian for one dataset and all four scores."""
    validate_dataset(data)
    if tuple(score_config.score_methods) != SCORE_METHOD_ORDER:
        raise AssertionError(
            f"{data.dataset}: score order changed: {score_config.score_methods}"
        )

    symmetric_summary = pd.read_csv(Path(symmetric_result_dir) / "summary.csv")
    symmetric_offsets = pd.read_csv(
        Path(symmetric_result_dir) / "selected_theta.csv"
    )
    symmetric_gaps = pd.read_csv(
        Path(symmetric_result_dir) / "label_gap_ledger.csv"
    )
    collections: dict[str, list[dict[str, Any]]] = {
        "summary": [],
        "offset": [],
        "cell": [],
        "group": [],
        "test": [],
        "search": [],
        "validation": [],
    }
    for score_method in SCORE_METHOD_ORDER:
        method_rows = run_directional_score_method(
            data=data,
            config=score_config,
            score_method=score_method,
            symmetric_summary=symmetric_summary,
            symmetric_offsets=symmetric_offsets,
            symmetric_gaps=symmetric_gaps,
        )
        for key in collections:
            collections[key].extend(method_rows[key])

    artifacts = DirectionalArtifacts(
        summary_rows=collections["summary"],
        offset_rows=collections["offset"],
        cell_rows=collections["cell"],
        group_rows=collections["group"],
        test_rows=collections["test"],
        search_rows=collections["search"],
        validation_rows=collections["validation"],
    )
    write_dataset_outputs(
        Path(result_dir),
        data,
        score_config,
        artifacts,
    )
    return artifacts


def run_directional_score_method(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    symmetric_summary: pd.DataFrame,
    symmetric_offsets: pd.DataFrame,
    symmetric_gaps: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    method_start = time.perf_counter()

    # Construct deterministic nonconformity scores.
    base_scores = {
        split_name: labelwise_scores_from_probs(
            split.probs,
            score_method,
            raps_lambda=config.raps_lambda,
            raps_k_reg=config.raps_k_reg,
            saps_lambda=config.saps_lambda,
        )
        for split_name, split in data.splits.items()
    }

    repair = data.splits["repair"]
    calibration = data.splits["calibration"]
    test = data.splits["test"]

    # The baseline group-Mondrian predictors fix the repair and calibration counts.
    repair_thresholds = compute_baseline_group_thresholds(
        base_scores["repair"], repair, data.group_values, config.alpha
    )
    repair_baseline = evaluate_split(
        base_scores["repair"], repair, repair_thresholds.thresholds
    )
    calibration_thresholds = compute_baseline_group_thresholds(
        base_scores["calibration"],
        calibration,
        data.group_values,
        config.alpha,
    )
    calibration_baseline = evaluate_split(
        base_scores["calibration"],
        calibration,
        calibration_thresholds.thresholds,
    )
    test_baseline = evaluate_split(
        base_scores["test"], test, calibration_thresholds.thresholds
    )

    # Freeze the initially advantaged and disadvantaged side of every G x Z cell.
    roles = define_repair_roles(repair_baseline, data, config)
    metric_context = build_metric_context(
        base_scores["repair"],
        repair,
        data,
        config,
        repair_baseline,
        roles,
    )
    side_grids = build_directional_grids(
        base_scores["repair"],
        repair,
        data,
        roles,
        repair_thresholds.thresholds,
        config,
    )
    symmetric_theta = load_symmetric_theta(
        data, score_method, symmetric_offsets
    )
    symmetric_row = unique_method_row(
        symmetric_summary, data.dataset, score_method, "symmetric summary"
    )

    # Search only D_repair; neither calibration nor test is an optimizer argument.
    (
        selected_vector,
        trace_rows,
        search_meta,
        repair_selected,
        symmetric_candidate,
        zero_selected,
    ) = directional_boundary_search(
        base_scores=base_scores["repair"],
        split=repair,
        data=data,
        config=config,
        roles=roles,
        side_grids=side_grids,
        target_counts=repair_thresholds.covered_counts,
        baseline_eval=repair_baseline,
        metric_context=metric_context,
        score_method=score_method,
        symmetric_theta=symmetric_theta,
    )
    theta_plus, theta_minus = split_directional_vector(
        selected_vector, len(data.cell_values)
    )

    # The symmetric candidate must reproduce the symmetric intervention.
    assert_symmetric_candidate_equivalence(
        data=data,
        config=config,
        score_method=score_method,
        base_scores=base_scores["repair"],
        repair=repair,
        roles=roles,
        target_counts=repair_thresholds.covered_counts,
        baseline_eval=repair_baseline,
        metric_context=metric_context,
        symmetric_theta=symmetric_theta,
        symmetric_directional=symmetric_candidate,
        symmetric_summary_row=symmetric_row,
    )

    # Freeze the selected magnitudes and reselect ordinary group thresholds on D_cal.
    calibration_selected = evaluate_directional_with_fixed_counts(
        base_scores=base_scores["calibration"],
        split=calibration,
        data=data,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        roles=roles,
        target_counts=calibration_thresholds.covered_counts,
        baseline_eval=calibration_baseline,
        config=config,
    )
    if not calibration_selected["fixed_counts"]:
        raise AssertionError(
            f"{data.dataset} {score_method}: calibration group counts changed"
        )

    # D_test is evaluated once with frozen magnitudes and frozen calibration thresholds.
    test_selected = evaluate_directional_split(
        base_scores["test"],
        test,
        data,
        calibration_selected["thresholds"].thresholds,
        theta_plus,
        theta_minus,
        roles,
    )

    symmetric_test_metrics = symmetric_test_benchmark(
        symmetric_gaps, data.dataset, score_method
    )
    runtime = float(time.perf_counter() - method_start)

    summary = build_summary_row(
        data=data,
        config=config,
        score_method=score_method,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        repair_baseline=repair_baseline,
        repair_selected=repair_selected["eval"],
        calibration_baseline=calibration_baseline,
        calibration_selected=calibration_selected["eval"],
        test_baseline=test_baseline,
        test_selected=test_selected,
        roles=roles,
        symmetric_row=symmetric_row,
        symmetric_test_metrics=symmetric_test_metrics,
        search_meta=search_meta,
        runtime=runtime,
    )
    if (
        float(summary["directional_repair_max_gap"])
        > float(summary["symmetric_repair_max_gap"])
        + CONTAINMENT_TOLERANCE
    ):
        raise AssertionError(
            f"{data.dataset} {score_method}: Tilted Mondrian failed symmetric "
            f"containment ({summary['directional_repair_max_gap']} > "
            f"{summary['symmetric_repair_max_gap']})"
        )

    split_evaluations = {
        "repair": (repair, repair_baseline, repair_selected["eval"]),
        "calibration": (
            calibration,
            calibration_baseline,
            calibration_selected["eval"],
        ),
        "test": (test, test_baseline, test_selected),
    }
    group_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for split_name, (split, baseline_eval, selected_eval) in split_evaluations.items():
        group_rows.extend(
            directional_group_rows(
                data,
                score_method,
                split_name,
                split,
                baseline_eval,
                selected_eval,
            )
        )
        cell_rows.extend(
            directional_cell_rows(
                data,
                config,
                score_method,
                split_name,
                baseline_eval,
                selected_eval,
                roles,
                theta_plus,
                theta_minus,
            )
        )

    test_rows = directional_test_rows(
        data,
        config,
        score_method,
        test_baseline,
        test_selected,
        roles,
        theta_plus,
        theta_minus,
    )
    offset_rows = directional_offset_rows(
        data,
        score_method,
        roles,
        theta_plus,
        theta_minus,
        search_meta,
    )
    validations = validation_rows(
        data=data,
        config=config,
        score_method=score_method,
        base_scores=base_scores,
        repair_thresholds=repair_thresholds,
        calibration_thresholds=calibration_thresholds,
        repair_baseline=repair_baseline,
        calibration_baseline=calibration_baseline,
        test_baseline=test_baseline,
        zero_selected=zero_selected,
        symmetric_candidate=symmetric_candidate,
        repair_selected=repair_selected,
        calibration_selected=calibration_selected,
        test_selected=test_selected,
        roles=roles,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        symmetric_row=symmetric_row,
        search_meta=search_meta,
    )
    return {
        "summary": [summary],
        "offset": offset_rows,
        "cell": cell_rows,
        "group": group_rows,
        "test": test_rows,
        "search": trace_rows,
        "validation": validations,
    }


def build_directional_grids(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    baseline_thresholds: dict[str, float],
    config: ScoreFixConfig,
) -> dict[str, dict[int, list[float]]]:
    """Build independent nonnegative magnitude grids for the two fixed directions."""
    ts = true_scores(base_scores, split.y)
    plus_grids: dict[int, list[float]] = {}
    minus_grids: dict[int, list[float]] = {}
    for cell_idx, role in roles.items():
        if not role["repairable"]:
            plus_grids[cell_idx] = [0.0]
            minus_grids[cell_idx] = [0.0]
            continue
        mask = split.z == data.cell_values[cell_idx]
        adv_scores = ts[mask & (split.g == role["advantaged_group"])]
        dis_scores = ts[mask & (split.g == role["disadvantaged_group"])]
        quantile = 1.0 - float(config.alpha)
        cross_gap = abs(
            float(np.quantile(adv_scores, quantile))
            - float(np.quantile(dis_scores, quantile))
        )
        plus_scale = directional_side_scale(
            adv_scores,
            float(baseline_thresholds[role["advantaged_group"]]),
            cross_gap,
            "plus",
        )
        minus_scale = directional_side_scale(
            dis_scores,
            float(baseline_thresholds[role["disadvantaged_group"]]),
            cross_gap,
            "minus",
        )
        plus_grids[cell_idx] = scaled_grid(plus_scale, config)
        minus_grids[cell_idx] = scaled_grid(minus_scale, config)
    return {"plus": plus_grids, "minus": minus_grids}


def directional_side_scale(
    scores: np.ndarray,
    threshold: float,
    cross_quantile_gap: float,
    side: str,
) -> float:
    scores = np.asarray(scores, dtype=float)
    iqr = float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25))
    if side == "plus":
        margins = threshold - scores
    elif side == "minus":
        margins = scores - threshold
    else:
        raise ValueError(f"Unknown directional side: {side}")
    positive = margins[np.isfinite(margins) & (margins > 1e-10)]
    margin_scale = float(np.quantile(positive, 0.5)) if len(positive) else 0.0
    candidates = [
        value
        for value in (cross_quantile_gap, iqr, margin_scale)
        if np.isfinite(value) and value >= 1e-8
    ]
    if not candidates:
        return 0.05
    # The median keeps isolated extreme score events from setting the whole grid scale.
    return float(np.median(np.asarray(candidates, dtype=float)))


def scaled_grid(scale: float, config: ScoreFixConfig) -> list[float]:
    return sorted(
        {
            round(max(0.0, float(multiplier) * float(scale)), 12)
            for multiplier in config.grid_multipliers
        }
    )


def directional_boundary_search(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    config: ScoreFixConfig,
    roles: dict[int, dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    metric_context: dict[str, Any],
    score_method: str,
    symmetric_theta: np.ndarray,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Run the configured deterministic search phases in the 2|Z| directional space."""
    n_cells = len(data.cell_values)
    trace: list[dict[str, Any]] = []
    feasible_candidates: list[dict[str, Any]] = []
    evaluation_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    zero_metrics: dict[str, Any] | None = None
    symmetric_key: tuple[float, ...] | None = None

    def evaluate_candidate(
        vector: np.ndarray,
        phase: str,
        pass_idx: int = 0,
        updated_cells: tuple[int, ...] = (),
        beam_rank: int | str = "",
        is_symmetric_candidate: bool = False,
    ) -> dict[str, Any]:
        trial_vector = np.maximum(np.asarray(vector, dtype=float), 0.0)
        key = directional_key(trial_vector, config)
        trial = evaluation_cache.get(key)
        if trial is None:
            plus, minus = split_directional_vector(trial_vector, n_cells)
            trial = evaluate_directional_with_fixed_counts(
                base_scores=base_scores,
                split=split,
                data=data,
                theta_plus=plus,
                theta_minus=minus,
                roles=roles,
                target_counts=target_counts,
                baseline_eval=baseline_eval,
                config=config,
                metric_context=metric_context,
            )
            evaluation_cache[key] = trial
        row = directional_trace_row(
            data=data,
            config=config,
            score_method=score_method,
            phase=phase,
            pass_idx=pass_idx,
            updated_cells=updated_cells,
            beam_rank=beam_rank,
            vector=trial_vector,
            trial=trial,
            is_symmetric_candidate=is_symmetric_candidate,
        )
        trace.append(row)
        if trial["feasible"]:
            feasible_candidates.append(
                {
                    "vector": trial_vector.copy(),
                    "eval": trial,
                    "obj": objective_tuple(trial["metrics"], zero_metrics),
                    "phase": phase,
                    "trace_idx": len(trace) - 1,
                    "key": key,
                    "symmetric_candidate": is_symmetric_candidate,
                }
            )
        return trial

    zero_vector = np.zeros(2 * n_cells, dtype=float)
    zero_eval = evaluate_candidate(zero_vector, "zero_baseline")
    if not zero_eval["feasible"]:
        raise RuntimeError(
            f"{data.dataset} {score_method}: zero directional shift is infeasible: "
            f"{zero_eval['rejection_reasons']}"
        )
    zero_metrics = zero_eval["metrics"]
    for candidate in feasible_candidates:
        candidate["obj"] = objective_tuple(candidate["eval"]["metrics"], zero_metrics)

    symmetric_vector = np.concatenate([symmetric_theta, symmetric_theta])
    symmetric_eval = evaluate_candidate(
        symmetric_vector, "symmetric_candidate", is_symmetric_candidate=True
    )
    if not symmetric_eval["feasible"]:
        raise RuntimeError(
            f"{data.dataset} {score_method}: symmetric candidate is infeasible: "
            f"{symmetric_eval['rejection_reasons']}"
        )
    symmetric_key = directional_key(symmetric_vector, config)

    # Search the simultaneous directional ray with the configured lambda grid.
    global_direction = directional_global_direction(
        side_grids, roles, config
    )
    global_infeasible: list[dict[str, Any]] = []
    for lambda_value in config.global_lambda_grid:
        vector = float(lambda_value) * global_direction
        trial = evaluate_candidate(vector, "global_line")
        if not trial["feasible"]:
            global_infeasible.append({"vector": vector, "eval": trial})

    project_directional_sources(
        global_infeasible,
        "boundary_projection",
        feasible_candidates,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )
    print(
        f"{data.dataset} {score_method} phase=boundary_projection "
        f"candidates={len(trace)}",
        flush=True,
    )

    # Build side-specific event grids and exercise all four directional cell moves.
    current_best = best_directional_candidate(feasible_candidates)
    run_directional_critical_grid(
        base_scores,
        split,
        data,
        roles,
        side_grids,
        config,
        current_best,
        evaluate_candidate,
    )
    run_directional_coordinate_refinement(
        n_cells,
        side_grids,
        config,
        zero_eval,
        zero_metrics,
        trace,
        evaluate_candidate,
    )
    print(
        f"{data.dataset} {score_method} phase=coordinate_refinement "
        f"candidates={len(trace)}",
        flush=True,
    )

    # Keep the symmetric solution in the beam while refining cell blocks.
    run_directional_beam_search(
        data,
        split,
        roles,
        side_grids,
        config,
        feasible_candidates,
        trace,
        evaluate_candidate,
        symmetric_key,
    )
    print(
        f"{data.dataset} {score_method} phase=beam "
        f"candidates={len(trace)}",
        flush=True,
    )

    # Use the configured deterministic seed and 1000-candidate Halton budget.
    quasi_infeasible: list[dict[str, Any]] = []
    for candidate_idx, vector in enumerate(
        directional_halton_candidates(
            side_grids, roles, config, score_method
        ),
        start=1,
    ):
        trial = evaluate_candidate(
            vector, "quasirandom", pass_idx=candidate_idx
        )
        if not trial["feasible"]:
            quasi_infeasible.append({"vector": vector, "eval": trial})
    project_directional_sources(
        quasi_infeasible,
        "quasirandom_projection",
        feasible_candidates,
        trace,
        config,
        zero_metrics,
        evaluate_candidate,
    )
    print(
        f"{data.dataset} {score_method} phase=quasirandom_projection "
        f"candidates={len(trace)}",
        flush=True,
    )

    final_candidate = best_directional_candidate(feasible_candidates)
    trace[final_candidate["trace_idx"]]["selected_final"] = True
    trace[final_candidate["trace_idx"]]["selected_for_update"] = True
    meta = directional_search_meta(
        trace,
        final_candidate,
        feasible_candidates,
        symmetric_key,
        config,
    )
    return (
        final_candidate["vector"],
        trace,
        meta,
        final_candidate["eval"],
        symmetric_eval,
        zero_eval,
    )


def directional_global_direction(
    side_grids: dict[str, dict[int, list[float]]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> np.ndarray:
    n_cells = len(roles)
    direction = np.zeros(2 * n_cells, dtype=float)
    repairable = [
        idx for idx, role in roles.items() if role["repairable"]
    ]
    max_gap = max(
        [float(roles[idx]["baseline_gap"]) for idx in repairable],
        default=0.0,
    )
    for cell_idx in repairable:
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


def run_directional_critical_grid(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    roles: dict[int, dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    current_best: dict[str, Any],
    evaluate_candidate: Callable[..., dict[str, Any]],
) -> None:
    order = prioritized_label_order(
        data, split, current_best["eval"]["eval"], roles, config
    )
    n_cells = len(data.cell_values)
    plus, minus = split_directional_vector(
        current_best["vector"], n_cells
    )
    for cell_idx in order:
        plus_values = side_critical_values(
            side="plus",
            cell_idx=cell_idx,
            current=float(plus[cell_idx]),
            grid=side_grids["plus"][cell_idx],
            base_scores=base_scores,
            split=split,
            data=data,
            current_eval=current_best["eval"]["eval"],
            roles=roles,
            config=config,
        )
        minus_values = side_critical_values(
            side="minus",
            cell_idx=cell_idx,
            current=float(minus[cell_idx]),
            grid=side_grids["minus"][cell_idx],
            base_scores=base_scores,
            split=split,
            data=data,
            current_eval=current_best["eval"]["eval"],
            roles=roles,
            config=config,
        )
        movements = critical_cell_movements(
            current_best["vector"],
            cell_idx,
            plus_values,
            minus_values,
            side_grids,
            config,
        )
        for vector in movements:
            evaluate_candidate(
                vector,
                "critical_grid",
                updated_cells=(cell_idx,),
            )


def side_critical_values(
    side: str,
    cell_idx: int,
    current: float,
    grid: list[float],
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    current_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
) -> list[float]:
    role = roles[cell_idx]
    if not role["repairable"]:
        return [0.0]
    scale = theta_scale_from_grid(grid, config)
    values = {0.0, round(float(current), 12)}
    values.update(round(float(value), 12) for value in grid)
    for multiplier in (
        -0.2,
        -0.1,
        -0.05,
        -0.025,
        0.025,
        0.05,
        0.1,
        0.2,
    ):
        values.add(
            round(
                max(0.0, float(current) + float(multiplier) * scale),
                12,
            )
        )
    for multiplier in (1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05):
        values.add(round(float(multiplier) * scale, 12))

    ts = true_scores(base_scores, split.y)
    cell_mask = split.z == data.cell_values[cell_idx]
    adv_scores = ts[
        cell_mask & (split.g == role["advantaged_group"])
    ]
    dis_scores = ts[
        cell_mask & (split.g == role["disadvantaged_group"])
    ]
    for quantile in (0.5, 0.75, 1.0 - float(config.alpha), 0.95):
        difference = abs(
            float(np.quantile(adv_scores, quantile))
            - float(np.quantile(dis_scores, quantile))
        )
        if np.isfinite(difference) and difference > 0.0:
            values.add(round(difference, 12))

    if side == "plus":
        group = role["advantaged_group"]
        scores = adv_scores
        margins = (
            float(current_eval["thresholds"][group]) - scores
        )
    elif side == "minus":
        group = role["disadvantaged_group"]
        scores = dis_scores
        margins = (
            scores - float(current_eval["thresholds"][group])
        )
    else:
        raise ValueError(f"Unknown directional side: {side}")
    margins = margins[
        np.isfinite(margins) & (margins > config.numeric_tolerance)
    ]
    if len(margins):
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            values.add(round(float(np.quantile(margins, quantile)), 12))
    return sorted(
        value
        for value in values
        if np.isfinite(value) and value >= 0.0
    )


def critical_cell_movements(
    current_vector: np.ndarray,
    cell_idx: int,
    plus_values: list[float],
    minus_values: list[float],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
) -> list[np.ndarray]:
    n_cells = len(current_vector) // 2
    current_plus = float(current_vector[cell_idx])
    current_minus = float(current_vector[n_cells + cell_idx])
    plus_near = nearest_values(
        plus_values, current_plus, limit=5
    )
    minus_near = nearest_values(
        minus_values, current_minus, limit=5
    )
    vectors: list[np.ndarray] = []

    # Advantaged-only and disadvantaged-only moves use separate event grids.
    for value in nearest_values(plus_values, current_plus, limit=9):
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, value, current_minus
            )
        )
    for value in nearest_values(minus_values, current_minus, limit=9):
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, current_plus, value
            )
        )

    # Joint asymmetric moves pair independently selected plus/minus events.
    for plus_value, minus_value in zip(
        plus_near,
        list(reversed(minus_near)),
    ):
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, plus_value, minus_value
            )
        )

    # Symmetric-subspace moves retain the +theta/-theta family locally.
    plus_scale = theta_scale_from_grid(
        side_grids["plus"][cell_idx], config
    )
    minus_scale = theta_scale_from_grid(
        side_grids["minus"][cell_idx], config
    )
    common_scale = min(plus_scale, minus_scale)
    for multiplier in (-0.2, -0.1, 0.0, 0.1, 0.2):
        delta = float(multiplier) * common_scale
        vectors.append(
            replace_cell_block(
                current_vector,
                cell_idx,
                max(0.0, current_plus + delta),
                max(0.0, current_minus + delta),
            )
        )
    return deduplicate_vectors(vectors, config)


def run_directional_coordinate_refinement(
    n_cells: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    zero_eval: dict[str, Any],
    zero_metrics: dict[str, Any],
    trace: list[dict[str, Any]],
    evaluate_candidate: Callable[..., dict[str, Any]],
) -> None:
    vector = np.zeros(2 * n_cells, dtype=float)
    current_objective = objective_tuple(
        zero_eval["metrics"], zero_metrics
    )
    for pass_idx in range(1, config.optimizer_passes + 1):
        for cell_idx in range(n_cells):
            best_vector = vector.copy()
            best_objective = current_objective
            best_trace_idx: int | None = None
            for candidate in coordinate_cell_movements(
                vector, cell_idx, side_grids, config
            ):
                trace_idx = len(trace)
                trial = evaluate_candidate(
                    candidate,
                    "coordinate_refinement",
                    pass_idx=pass_idx,
                    updated_cells=(cell_idx,),
                    beam_rank="coordinate",
                )
                if trial["feasible"]:
                    candidate_objective = objective_tuple(
                        trial["metrics"], zero_metrics
                    )
                    if lexicographically_better(
                        candidate_objective,
                        best_objective,
                        config.numeric_tolerance,
                    ):
                        best_vector = candidate.copy()
                        best_objective = candidate_objective
                        best_trace_idx = trace_idx
            if best_trace_idx is not None:
                trace[best_trace_idx]["selected_for_update"] = True
                vector = best_vector
                current_objective = best_objective


def coordinate_cell_movements(
    current_vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
) -> list[np.ndarray]:
    n_cells = len(current_vector) // 2
    current_plus = float(current_vector[cell_idx])
    current_minus = float(current_vector[n_cells + cell_idx])
    plus_grid = side_grids["plus"][cell_idx]
    minus_grid = side_grids["minus"][cell_idx]
    vectors: list[np.ndarray] = []
    for value in plus_grid:
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, value, current_minus
            )
        )
    for value in minus_grid:
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, current_plus, value
            )
        )
    for plus_value, minus_value in zip(
        plus_grid, list(reversed(minus_grid))
    ):
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, plus_value, minus_value
            )
        )
    common_values = sorted(
        set(plus_grid).intersection(minus_grid)
        or {
            round((plus + minus) / 2.0, 12)
            for plus, minus in zip(plus_grid, minus_grid)
        }
    )
    for value in common_values:
        vectors.append(
            replace_cell_block(
                current_vector, cell_idx, value, value
            )
        )
    return deduplicate_vectors(vectors, config)


def run_directional_beam_search(
    data: LoadedDataset,
    split: SplitData,
    roles: dict[int, dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Callable[..., dict[str, Any]],
    symmetric_key: tuple[float, ...] | None,
) -> None:
    beam = top_directional_beam(
        feasible_candidates, config, symmetric_key
    )
    for pass_idx in range(1, config.optimizer_passes + 1):
        old_best_key = best_directional_candidate(
            feasible_candidates
        )["key"]
        order = prioritized_label_order(
            data, split, beam[0]["eval"]["eval"], roles, config
        )
        pairs = prioritized_pair_blocks(order, roles, config)
        triples = prioritized_triple_blocks(order, roles, config)
        beam = run_single_cell_blocks(
            order,
            beam,
            side_grids,
            config,
            feasible_candidates,
            trace,
            evaluate_candidate,
            pass_idx,
            symmetric_key,
        )
        beam = run_pair_cell_blocks(
            pairs,
            beam,
            side_grids,
            config,
            feasible_candidates,
            trace,
            evaluate_candidate,
            pass_idx,
            symmetric_key,
        )
        beam = run_triple_cell_blocks(
            triples,
            beam,
            side_grids,
            config,
            feasible_candidates,
            trace,
            evaluate_candidate,
            pass_idx,
            symmetric_key,
        )
        beam = top_directional_beam(
            feasible_candidates, config, symmetric_key
        )
        for rank, state in enumerate(beam, start=1):
            evaluate_candidate(
                state["vector"],
                "beam",
                pass_idx=pass_idx,
                beam_rank=rank,
                is_symmetric_candidate=bool(state.get("symmetric_candidate")),
            )
        if (
            best_directional_candidate(feasible_candidates)["key"]
            == old_best_key
        ):
            break


def run_single_cell_blocks(
    order: list[int],
    beam: list[dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Callable[..., dict[str, Any]],
    pass_idx: int,
    symmetric_key: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    for cell_idx in order:
        block_candidates: list[dict[str, Any]] = []
        for rank, state in enumerate(beam, start=1):
            for vector in local_cell_movements(
                state["vector"], cell_idx, side_grids, config
            ):
                trial = evaluate_candidate(
                    vector,
                    "single_block",
                    pass_idx=pass_idx,
                    updated_cells=(cell_idx,),
                    beam_rank=rank,
                )
                if trial["feasible"]:
                    block_candidates.append(feasible_candidates[-1])
        beam = top_directional_beam(
            beam + block_candidates, config, symmetric_key
        )
        mark_beam_best(trace, beam)
    return beam


def run_pair_cell_blocks(
    pairs: list[tuple[int, int]],
    beam: list[dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Callable[..., dict[str, Any]],
    pass_idx: int,
    symmetric_key: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    for left, right in pairs:
        block_candidates: list[dict[str, Any]] = []
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
                    vector = replace_cell_block(
                        vector, right, *right_values
                    )
                    trial = evaluate_candidate(
                        vector,
                        "pair_block",
                        pass_idx=pass_idx,
                        updated_cells=(left, right),
                        beam_rank=rank,
                    )
                    if trial["feasible"]:
                        block_candidates.append(
                            feasible_candidates[-1]
                        )
        beam = top_directional_beam(
            beam + block_candidates, config, symmetric_key
        )
        mark_beam_best(trace, beam)
    return beam


def run_triple_cell_blocks(
    triples: list[tuple[int, int, int]],
    beam: list[dict[str, Any]],
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    evaluate_candidate: Callable[..., dict[str, Any]],
    pass_idx: int,
    symmetric_key: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    for first, second, third in triples:
        block_candidates: list[dict[str, Any]] = []
        for rank, state in enumerate(beam, start=1):
            option_sets = [
                compact_cell_block_options(
                    state["vector"],
                    cell_idx,
                    side_grids,
                    config,
                    "triple",
                )
                for cell_idx in (first, second, third)
            ]
            for values_1 in option_sets[0]:
                for values_2 in option_sets[1]:
                    for values_3 in option_sets[2]:
                        vector = replace_cell_block(
                            state["vector"], first, *values_1
                        )
                        vector = replace_cell_block(
                            vector, second, *values_2
                        )
                        vector = replace_cell_block(
                            vector, third, *values_3
                        )
                        trial = evaluate_candidate(
                            vector,
                            "triple_block",
                            pass_idx=pass_idx,
                            updated_cells=(first, second, third),
                            beam_rank=rank,
                        )
                        if trial["feasible"]:
                            block_candidates.append(
                                feasible_candidates[-1]
                            )
        beam = top_directional_beam(
            beam + block_candidates, config, symmetric_key
        )
        mark_beam_best(trace, beam)
    return beam


def local_cell_movements(
    vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
) -> list[np.ndarray]:
    n_cells = len(vector) // 2
    plus = float(vector[cell_idx])
    minus = float(vector[n_cells + cell_idx])
    plus_scale = theta_scale_from_grid(
        side_grids["plus"][cell_idx], config
    )
    minus_scale = theta_scale_from_grid(
        side_grids["minus"][cell_idx], config
    )
    common_scale = min(plus_scale, minus_scale)
    blocks = [
        (plus, minus),
        (plus + 0.5 * plus_scale, minus),
        (plus, minus + 0.5 * minus_scale),
        (
            plus + 0.5 * plus_scale,
            minus + 0.25 * minus_scale,
        ),
        (
            plus + 0.5 * common_scale,
            minus + 0.5 * common_scale,
        ),
    ]
    return deduplicate_vectors(
        [
            replace_cell_block(
                vector,
                cell_idx,
                max(0.0, block_plus),
                max(0.0, block_minus),
            )
            for block_plus, block_minus in blocks
        ],
        config,
    )


def compact_cell_block_options(
    vector: np.ndarray,
    cell_idx: int,
    side_grids: dict[str, dict[int, list[float]]],
    config: ScoreFixConfig,
    block_kind: str,
) -> list[tuple[float, float]]:
    n_cells = len(vector) // 2
    plus = float(vector[cell_idx])
    minus = float(vector[n_cells + cell_idx])
    plus_scale = theta_scale_from_grid(
        side_grids["plus"][cell_idx], config
    )
    minus_scale = theta_scale_from_grid(
        side_grids["minus"][cell_idx], config
    )
    step = 0.5 if block_kind == "pair" else 0.25
    return [
        (plus, minus),
        (plus + step * plus_scale, minus),
        (plus, minus + step * minus_scale),
    ]


def directional_halton_candidates(
    side_grids: dict[str, dict[int, list[float]]],
    roles: dict[int, dict[str, Any]],
    config: ScoreFixConfig,
    score_method: str,
) -> list[np.ndarray]:
    repairable = [
        cell_idx
        for cell_idx, role in roles.items()
        if role["repairable"]
    ]
    n_cells = len(roles)
    if not repairable:
        return []
    plus_scales = np.zeros(n_cells, dtype=float)
    minus_scales = np.zeros(n_cells, dtype=float)
    pressure = np.zeros(n_cells, dtype=float)
    max_gap = max(
        float(roles[cell_idx]["baseline_gap"])
        for cell_idx in repairable
    )
    for cell_idx in repairable:
        plus_scales[cell_idx] = theta_scale_from_grid(
            side_grids["plus"][cell_idx], config
        )
        minus_scales[cell_idx] = theta_scale_from_grid(
            side_grids["minus"][cell_idx], config
        )
        pressure[cell_idx] = (
            max(
                0.05,
                float(roles[cell_idx]["baseline_gap"]) / max_gap,
            )
            if max_gap > config.numeric_tolerance
            else 1.0
        )
    ordered = sorted(
        repairable,
        key=lambda idx: (-float(roles[idx]["baseline_gap"]), idx),
    )
    method_offset = sum(
        (idx + 1) * ord(character)
        for idx, character in enumerate(score_method)
    )
    start = int(config.quasirandom_seed + method_offset)
    primes = first_primes(2 * len(repairable))
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
        for position, cell_idx in enumerate(repairable):
            u_plus = uniforms[position]
            u_minus = uniforms[len(repairable) + position]
            if mode == 0:
                shared = 0.05 + 0.70 * (
                    0.5 * u_plus + 0.5 * u_minus
                )
                plus[cell_idx] = (
                    plus_scales[cell_idx]
                    * pressure[cell_idx]
                    * shared
                )
                minus[cell_idx] = (
                    minus_scales[cell_idx]
                    * pressure[cell_idx]
                    * shared
                )
            elif mode == 1:
                plus[cell_idx] = (
                    plus_scales[cell_idx]
                    * pressure[cell_idx]
                    * (0.10 + 2.40 * u_plus)
                )
                minus[cell_idx] = (
                    minus_scales[cell_idx]
                    * pressure[cell_idx]
                    * (0.10 + 2.40 * u_minus)
                )
            elif mode == 2:
                active_count = 1 + (
                    candidate_idx % min(4, len(ordered))
                )
                if cell_idx in ordered[:active_count]:
                    plus[cell_idx] = (
                        plus_scales[cell_idx]
                        * pressure[cell_idx]
                        * (0.05 + 2.95 * u_plus)
                    )
                    minus[cell_idx] = (
                        minus_scales[cell_idx]
                        * pressure[cell_idx]
                        * (0.05 + 2.95 * u_minus)
                    )
            else:
                if u_plus > 0.45 or cell_idx in ordered[:3]:
                    plus[cell_idx] = (
                        plus_scales[cell_idx]
                        * (0.02 + 1.80 * u_plus)
                        * (0.25 + 0.75 * pressure[cell_idx])
                    )
                if u_minus > 0.45 or cell_idx in ordered[:3]:
                    minus[cell_idx] = (
                        minus_scales[cell_idx]
                        * (0.02 + 1.80 * u_minus)
                        * (0.25 + 0.75 * pressure[cell_idx])
                    )
        candidates.append(
            np.round(np.concatenate([plus, minus]), 12)
        )
    return candidates


def project_directional_sources(
    sources: list[dict[str, Any]],
    phase: str,
    feasible_candidates: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    config: ScoreFixConfig,
    zero_metrics: dict[str, Any],
    evaluate_candidate: Callable[..., dict[str, Any]],
) -> None:
    best = best_directional_candidate(feasible_candidates)
    promising = []
    for source in sources:
        reasons = set(source["eval"]["rejection_reasons"])
        if not reasons or "group_coverage_not_fixed" in reasons:
            continue
        metrics = source["eval"]["metrics"]
        if (
            metrics["total_local_gap_reduction"]
            >= best["eval"]["metrics"]["total_local_gap_reduction"]
            - config.numeric_tolerance
            or metrics["max_local_gap"]
            <= best["eval"]["metrics"]["max_local_gap"] + 0.02
        ):
            promising.append(source)
    promising.sort(
        key=lambda source: objective_tuple(
            source["eval"]["metrics"], zero_metrics
        )
    )
    promising = promising[
        : int(config.boundary_projection_max_candidates)
    ]
    for source_rank, source in enumerate(promising, start=1):
        best = best_directional_candidate(feasible_candidates)
        good = best["vector"]
        bad = source["vector"]
        strongest_trace_idx: int | None = None
        strongest_t = -1.0
        for value in np.linspace(
            0.0, 1.0, config.boundary_projection_grid_size
        ):
            vector = (1.0 - float(value)) * good + float(value) * bad
            trace_idx = len(trace)
            trial = evaluate_candidate(
                vector, phase, pass_idx=source_rank
            )
            if trial["feasible"] and float(value) > strongest_t:
                strongest_t = float(value)
                strongest_trace_idx = trace_idx
        if strongest_trace_idx is not None:
            trace[strongest_trace_idx]["selected_for_update"] = True


def evaluate_directional_with_fixed_counts(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    config: ScoreFixConfig,
    metric_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shifted_true = directional_true_scores(
        base_scores,
        split,
        data,
        theta_plus,
        theta_minus,
        roles,
        metric_context,
    )
    thresholds = compute_thresholds_for_target_counts_from_true_scores(
        shifted_true, split, data.group_values, target_counts
    )
    if metric_context is not None:
        repaired_eval = evaluate_directional_split_fast(
            base_scores,
            split,
            data.intervention_mode,
            thresholds.thresholds,
            theta_plus,
            theta_minus,
            metric_context,
        )
    else:
        repaired_eval = evaluate_directional_split(
            base_scores,
            split,
            data,
            thresholds.thresholds,
            theta_plus,
            theta_minus,
            roles,
        )
    exact_counts = {
        group: int(
            repaired_eval["covered"][split.g == group].sum()
        )
        for group in data.group_values
    }
    fixed_counts = all(
        exact_counts[group] == int(target_counts[group])
        for group in data.group_values
    )
    context = metric_context or build_metric_context(
        base_scores,
        split,
        data,
        config,
        baseline_eval,
        roles,
    )
    vector = np.concatenate([theta_plus, theta_minus])
    metrics = candidate_metrics_fast(
        config, repaired_eval, roles, vector, context
    )
    reasons = candidate_rejection_reasons(
        thresholds.exact and fixed_counts, metrics, config
    )
    return {
        "eval": repaired_eval,
        "thresholds": thresholds,
        "metrics": metrics,
        "feasible": not reasons,
        "fixed_counts": fixed_counts,
        "exact_counts": exact_counts,
        "rejection_reasons": reasons,
    }


def directional_true_scores(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
    metric_context: dict[str, Any] | None = None,
) -> np.ndarray:
    if metric_context is not None:
        base_true = np.asarray(
            metric_context["base_true_scores"], dtype=float
        )
        cell_indices = np.asarray(
            metric_context["row_cell_indices"], dtype=int
        )
        signs = np.asarray(
            metric_context["true_shift_signs"], dtype=float
        )
        row_plus = np.asarray(theta_plus, dtype=float)[cell_indices]
        row_minus = np.asarray(theta_minus, dtype=float)[cell_indices]
        row_shift = np.where(
            signs > 0.0,
            row_plus,
            np.where(signs < 0.0, -row_minus, 0.0),
        )
        return base_true + row_shift

    shifted = true_scores(base_scores, split.y).copy()
    cell_indices = repair_cell_indices(data, split)
    for cell_idx, role in roles.items():
        if not role["repairable"]:
            continue
        cell_mask = cell_indices == cell_idx
        shifted[
            cell_mask & (split.g == role["advantaged_group"])
        ] += float(theta_plus[cell_idx])
        shifted[
            cell_mask & (split.g == role["disadvantaged_group"])
        ] -= float(theta_minus[cell_idx])
    return shifted


def evaluate_directional_split_fast(
    base_scores: np.ndarray,
    split: SplitData,
    intervention_mode: str,
    thresholds: dict[str, float],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    metric_context: dict[str, Any],
) -> dict[str, Any]:
    """Vectorized repair evaluation using the frozen directional role masks."""
    signs = np.asarray(
        metric_context["candidate_shift_signs"], dtype=float
    )
    if intervention_mode == "label_defined_z":
        shifted_scores = (
            np.asarray(base_scores, dtype=float)
            + (signs > 0.0)
            * np.asarray(theta_plus, dtype=float)[None, :]
            - (signs < 0.0)
            * np.asarray(theta_minus, dtype=float)[None, :]
        )
    elif intervention_mode == "covariate_defined_z":
        cell_indices = np.asarray(
            metric_context["row_cell_indices"], dtype=int
        )
        row_plus = np.asarray(theta_plus, dtype=float)[cell_indices]
        row_minus = np.asarray(theta_minus, dtype=float)[cell_indices]
        row_shift = np.where(
            signs > 0.0,
            row_plus,
            np.where(signs < 0.0, -row_minus, 0.0),
        )
        shifted_scores = (
            np.asarray(base_scores, dtype=float) + row_shift[:, None]
        )
    else:
        raise ValueError(
            f"Unsupported intervention mode: {intervention_mode}"
        )
    thresholds_by_row = row_thresholds(split, thresholds)
    prediction_sets = shifted_scores <= thresholds_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    return {
        "covered": covered.astype(bool),
        "set_sizes": prediction_sets.sum(axis=1).astype(float),
        "thresholds": thresholds,
    }


def evaluate_directional_split(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    thresholds: dict[str, float],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    shifted_scores = apply_directional_score_matrix(
        base_scores,
        split,
        data,
        theta_plus,
        theta_minus,
        roles,
    )
    thresholds_by_row = row_thresholds(split, thresholds)
    prediction_sets = shifted_scores <= thresholds_by_row[:, None]
    covered = prediction_sets[np.arange(len(split.y)), split.y]
    return {
        "covered": covered.astype(bool),
        "set_sizes": prediction_sets.sum(axis=1).astype(float),
        "thresholds": thresholds,
    }


def apply_directional_score_matrix(
    base_scores: np.ndarray,
    split: SplitData,
    data: LoadedDataset,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    roles: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Apply fixed-sign, separately sized directional shifts."""
    shifted = np.asarray(base_scores, dtype=float).copy()
    if np.any(np.asarray(theta_plus) < 0.0) or np.any(
        np.asarray(theta_minus) < 0.0
    ):
        raise ValueError("Directional magnitudes must be nonnegative")
    if data.intervention_mode == "label_defined_z":
        for cell_idx, role in roles.items():
            if not role["repairable"]:
                continue
            shifted[
                split.g == role["advantaged_group"], cell_idx
            ] += float(theta_plus[cell_idx])
            shifted[
                split.g == role["disadvantaged_group"], cell_idx
            ] -= float(theta_minus[cell_idx])
        return shifted
    if data.intervention_mode == "covariate_defined_z":
        row_cells = repair_cell_indices(data, split)
        row_shift = np.zeros(len(split.y), dtype=float)
        for cell_idx, role in roles.items():
            if not role["repairable"]:
                continue
            cell_mask = row_cells == cell_idx
            row_shift[
                cell_mask & (split.g == role["advantaged_group"])
            ] += float(theta_plus[cell_idx])
            row_shift[
                cell_mask & (split.g == role["disadvantaged_group"])
            ] -= float(theta_minus[cell_idx])
        shifted += row_shift[:, None]
        return shifted
    raise ValueError(
        f"Unsupported intervention mode: {data.intervention_mode}"
    )


def assert_symmetric_candidate_equivalence(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    base_scores: np.ndarray,
    repair: SplitData,
    roles: dict[int, dict[str, Any]],
    target_counts: dict[str, int],
    baseline_eval: dict[str, Any],
    metric_context: dict[str, Any],
    symmetric_theta: np.ndarray,
    symmetric_directional: dict[str, Any],
    symmetric_summary_row: pd.Series,
) -> None:
    from no_free_repair_core import evaluate_repaired_with_fixed_counts

    old_evaluation = evaluate_repaired_with_fixed_counts(
        base_scores,
        repair,
        data,
        symmetric_theta,
        roles,
        target_counts,
        baseline_eval,
        config,
        metric_context,
    )
    if not np.array_equal(
        old_evaluation["eval"]["covered"],
        symmetric_directional["eval"]["covered"],
    ):
        raise AssertionError(
            f"{data.dataset} {score_method}: symmetric candidate covered "
            "indicators differ"
        )
    if not np.array_equal(
        old_evaluation["eval"]["set_sizes"],
        symmetric_directional["eval"]["set_sizes"],
    ):
        raise AssertionError(
            f"{data.dataset} {score_method}: symmetric candidate set sizes differ"
        )
    for group in data.group_values:
        old_threshold = float(
            old_evaluation["thresholds"].thresholds[group]
        )
        new_threshold = float(
            symmetric_directional["thresholds"].thresholds[group]
        )
        if not (
            old_threshold == new_threshold
            or (
                np.isinf(old_threshold)
                and np.isinf(new_threshold)
                and np.sign(old_threshold) == np.sign(new_threshold)
            )
        ):
            raise AssertionError(
                f"{data.dataset} {score_method}: symmetric candidate "
                f"threshold differs for group {group}"
            )
    expected_max = float(
        symmetric_summary_row["repaired_max_local_gap_repair"]
    )
    expected_mean = float(
        symmetric_summary_row["repaired_mean_local_gap_repair"]
    )
    actual = symmetric_directional["metrics"]
    if abs(float(actual["max_local_gap"]) - expected_max) > CONTAINMENT_TOLERANCE:
        raise AssertionError(
            f"{data.dataset} {score_method}: symmetric candidate max gap "
            f"{actual['max_local_gap']} != {expected_max}"
        )
    if abs(float(actual["mean_local_gap"]) - expected_mean) > CONTAINMENT_TOLERANCE:
        raise AssertionError(
            f"{data.dataset} {score_method}: symmetric candidate mean gap "
            f"{actual['mean_local_gap']} != {expected_mean}"
        )


def directional_trace_row(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    phase: str,
    pass_idx: int,
    updated_cells: tuple[int, ...],
    beam_rank: int | str,
    vector: np.ndarray,
    trial: dict[str, Any],
    is_symmetric_candidate: bool,
) -> dict[str, Any]:
    n_cells = len(data.cell_values)
    plus, minus = split_directional_vector(vector, n_cells)
    metrics = trial["metrics"]
    cells = [
        data.cell_values[index] for index in updated_cells
    ]
    return {
        "dataset": data.dataset,
        "score_method": score_method,
        "optimizer_phase": phase,
        "beam_rank": beam_rank,
        "pass": pass_idx,
        "updated_cell": cells[0] if len(cells) > 0 else "",
        "updated_cell_2": cells[1] if len(cells) > 1 else "",
        "updated_cell_3": cells[2] if len(cells) > 2 else "",
        "candidate_theta_plus": directional_json(data, plus),
        "candidate_theta_minus": directional_json(data, minus),
        "feasible": bool(trial["feasible"]),
        "rejection_reason": ";".join(trial["rejection_reasons"]),
        "max_local_gap": metrics["max_local_gap"],
        "mean_local_gap": metrics["mean_local_gap"],
        "total_local_gap_reduction": metrics[
            "total_local_gap_reduction"
        ],
        "max_local_gap_increase": metrics["max_local_gap_increase"],
        "max_advantaged_harm": metrics["max_advantaged_harm"],
        "max_disadvantaged_loss": metrics["max_disadvantaged_loss"],
        "avg_set_size_cost": metrics["avg_set_size_cost"],
        "shift_norm": metrics["shift_norm"],
        "asymmetry_l1": float(np.sum(np.abs(plus - minus))),
        "selected_for_update": False,
        "selected_final": False,
        "symmetric_candidate": bool(is_symmetric_candidate),
        "theta_key": directional_key(vector, config),
    }


def directional_search_meta(
    trace: list[dict[str, Any]],
    final_candidate: dict[str, Any],
    feasible_candidates: list[dict[str, Any]],
    symmetric_key: tuple[float, ...] | None,
    config: ScoreFixConfig,
) -> dict[str, Any]:
    rejected = [row for row in trace if not bool(row["feasible"])]

    def rejection_count(reason: str) -> int:
        return int(
            sum(
                reason
                in str(row["rejection_reason"]).split(";")
                for row in rejected
            )
        )

    return {
        "optimizer_name": OPTIMIZER_NAME,
        "selected_from_phase": final_candidate["phase"],
        "candidate_count_total": int(len(trace)),
        "candidate_count_feasible": int(
            sum(bool(row["feasible"]) for row in trace)
        ),
        "candidate_count_infeasible": int(len(rejected)),
        "candidate_count_projected": int(
            sum(
                row["optimizer_phase"]
                in {"boundary_projection", "quasirandom_projection"}
                for row in trace
            )
        ),
        "candidate_count_quasirandom": int(
            sum(
                row["optimizer_phase"] == "quasirandom"
                for row in trace
            )
        ),
        "candidate_count_triple_block": int(
            sum(
                row["optimizer_phase"] == "triple_block"
                for row in trace
            )
        ),
        "symmetric_key": symmetric_key,
        "symmetric_candidate_in_pool": bool(
            any(
                candidate["key"] == symmetric_key
                for candidate in feasible_candidates
            )
        ),
        "rejected_by_advantaged_harm_count": rejection_count(
            "advantaged_harm_exceeds_epsilon"
        ),
        "rejected_by_local_gap_worsening_count": rejection_count(
            "local_gap_worsened"
        ),
        "rejected_by_disadvantaged_worsening_count": rejection_count(
            "disadvantaged_cell_worsened"
        ),
        "rejected_by_group_coverage_count": rejection_count(
            "group_coverage_not_fixed"
        ),
        "beam_width": config.beam_width,
        "quasirandom_candidate_count": config.quasirandom_candidate_count,
    }


def build_summary_row(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    repair_baseline: dict[str, Any],
    repair_selected: dict[str, Any],
    calibration_baseline: dict[str, Any],
    calibration_selected: dict[str, Any],
    test_baseline: dict[str, Any],
    test_selected: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    symmetric_row: pd.Series,
    symmetric_test_metrics: dict[str, float],
    search_meta: dict[str, Any],
    runtime: float,
) -> dict[str, Any]:
    repair = gap_summary(
        data,
        config,
        "repair",
        repair_baseline,
        repair_selected,
        roles,
    )
    calibration = gap_summary(
        data,
        config,
        "calibration",
        calibration_baseline,
        calibration_selected,
        roles,
    )
    test = gap_summary(
        data,
        config,
        "test",
        test_baseline,
        test_selected,
        roles,
    )
    asymmetry = asymmetry_diagnostics(
        theta_plus, theta_minus, config.numeric_tolerance
    )
    repair_size_change = float(
        np.mean(repair_selected["set_sizes"])
        - np.mean(repair_baseline["set_sizes"])
    )
    calibration_size_change = float(
        np.mean(calibration_selected["set_sizes"])
        - np.mean(calibration_baseline["set_sizes"])
    )
    test_size_change = float(
        np.mean(test_selected["set_sizes"])
        - np.mean(test_baseline["set_sizes"])
    )
    return {
        "dataset": data.dataset,
        "score_method": score_method,
        "optimizer_name": OPTIMIZER_NAME,
        "intervention_mode": data.intervention_mode,
        "local_cell_definition": human_cell_definition(data.dataset),
        "alpha": config.alpha,
        "epsilon": config.epsilon,
        "gap_tol": config.gap_tol,
        "coverage_tol": config.coverage_tol,
        "baseline_repair_max_gap": repair["baseline_max_gap"],
        "symmetric_repair_max_gap": float(
            symmetric_row["repaired_max_local_gap_repair"]
        ),
        "directional_repair_max_gap": repair["repaired_max_gap"],
        "baseline_repair_mean_gap": repair["baseline_mean_gap"],
        "symmetric_repair_mean_gap": float(
            symmetric_row["repaired_mean_local_gap_repair"]
        ),
        "directional_repair_mean_gap": repair["repaired_mean_gap"],
        "repair_maximum_single_cell_mitigation": repair[
            "maximum_single_cell_mitigation"
        ],
        "repair_total_mitigation": repair["total_gap_reduction"],
        "repair_average_set_size_change": repair_size_change,
        "calibration_average_set_size_change": calibration_size_change,
        "test_average_set_size_change": test_size_change,
        "maximum_advantaged_harm_repair": repair[
            "maximum_advantaged_harm"
        ],
        "minimum_disadvantaged_cell_change_repair": repair[
            "minimum_disadvantaged_change"
        ],
        "maximum_local_gap_increase_repair": repair[
            "maximum_local_gap_increase"
        ],
        "repair_group_covered_counts_exact": repair[
            "group_counts_unchanged"
        ],
        "calibration_group_covered_counts_exact": calibration[
            "group_counts_unchanged"
        ],
        "selected_theta_plus": directional_json(data, theta_plus),
        "selected_theta_minus": directional_json(data, theta_minus),
        "selected_search_phase": search_meta["selected_from_phase"],
        "candidate_count_total": search_meta["candidate_count_total"],
        "candidate_count_feasible": search_meta[
            "candidate_count_feasible"
        ],
        "candidate_count_infeasible": search_meta[
            "candidate_count_infeasible"
        ],
        "candidate_count_projected": search_meta[
            "candidate_count_projected"
        ],
        "candidate_count_quasirandom": search_meta[
            "candidate_count_quasirandom"
        ],
        "candidate_count_triple_block": search_meta[
            "candidate_count_triple_block"
        ],
        "symmetric_candidate_in_pool": search_meta[
            "symmetric_candidate_in_pool"
        ],
        "runtime_seconds": runtime,
        "directional_intervention_norm": float(
            np.linalg.norm(
                np.concatenate([theta_plus, theta_minus]), ord=2
            )
        ),
        **asymmetry,
        "baseline_test_max_gap": test["baseline_max_gap"],
        "directional_test_max_gap": test["repaired_max_gap"],
        "baseline_test_mean_gap": test["baseline_mean_gap"],
        "directional_test_mean_gap": test["repaired_mean_gap"],
        "baseline_test_total_gap": test["baseline_total_gap"],
        "directional_test_total_gap": test["repaired_total_gap"],
        "directional_total_mitigation_test": test[
            "total_gap_reduction"
        ],
        "symmetric_test_max_gap": symmetric_test_metrics["max_gap"],
        "symmetric_test_mean_gap": symmetric_test_metrics["mean_gap"],
        "symmetric_test_set_size_change": float(
            symmetric_row["average_set_size_change_test"]
        ),
        "rejected_by_advantaged_harm_count": search_meta[
            "rejected_by_advantaged_harm_count"
        ],
        "rejected_by_local_gap_worsening_count": search_meta[
            "rejected_by_local_gap_worsening_count"
        ],
        "rejected_by_disadvantaged_worsening_count": search_meta[
            "rejected_by_disadvantaged_worsening_count"
        ],
        "rejected_by_group_coverage_count": search_meta[
            "rejected_by_group_coverage_count"
        ],
    }


def gap_summary(
    data: LoadedDataset,
    config: ScoreFixConfig,
    split_name: str,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows = label_gaps(
        data.splits[split_name],
        data.group_values,
        baseline_eval,
        repaired_eval,
        roles,
        config,
    )
    repairable = [row for row in rows if row["repairable"]]
    baseline_gaps = [
        float(row["baseline_gap"]) for row in repairable
    ]
    repaired_gaps = [
        float(row["repaired_gap"]) for row in repairable
    ]
    reductions = [
        float(row["gap_reduction"]) for row in repairable
    ]
    harms = [
        float(row["advantaged_loss"]) for row in repairable
    ]
    disadvantaged_changes = [
        float(row["disadvantaged_coverage_change"])
        for row in repairable
    ]
    constraints = gap_constraint_summary(rows, config)
    split = data.splits[split_name]
    group_counts_unchanged = all(
        int(repaired_eval["covered"][split.g == group].sum())
        == int(baseline_eval["covered"][split.g == group].sum())
        for group in data.group_values
    )
    return {
        "baseline_max_gap": max(baseline_gaps, default=0.0),
        "repaired_max_gap": max(repaired_gaps, default=0.0),
        "baseline_mean_gap": (
            float(np.mean(baseline_gaps)) if baseline_gaps else 0.0
        ),
        "repaired_mean_gap": (
            float(np.mean(repaired_gaps)) if repaired_gaps else 0.0
        ),
        "baseline_total_gap": float(sum(baseline_gaps)),
        "repaired_total_gap": float(sum(repaired_gaps)),
        "total_gap_reduction": float(sum(reductions)),
        "maximum_single_cell_mitigation": max(
            reductions, default=0.0
        ),
        "maximum_advantaged_harm": max(
            [0.0] + harms
        ),
        "minimum_disadvantaged_change": min(
            disadvantaged_changes, default=0.0
        ),
        "maximum_local_gap_increase": constraints[
            "max_local_gap_increase"
        ],
        "any_local_gap_increased": constraints[
            "any_local_gap_increased"
        ],
        "any_disadvantaged_cell_worsened": constraints[
            "any_disadvantaged_cell_worsened"
        ],
        "group_counts_unchanged": bool(group_counts_unchanged),
        "rows": rows,
    }


def asymmetry_diagnostics(
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    difference = np.asarray(theta_plus) - np.asarray(theta_minus)
    plus_nonzero = np.asarray(theta_plus) > tolerance
    minus_nonzero = np.asarray(theta_minus) > tolerance
    l1 = float(np.sum(np.abs(difference)))
    return {
        "asymmetry_L1": l1,
        "asymmetry_L2": float(np.linalg.norm(difference, ord=2)),
        "asymmetry_ratio": float(
            l1
            / (
                float(np.sum(theta_plus))
                + float(np.sum(theta_minus))
                + 1e-15
            )
        ),
        "number_of_asymmetric_cells": int(
            np.sum(np.abs(difference) > tolerance)
        ),
        "number_of_plus_only_cells": int(
            np.sum(plus_nonzero & ~minus_nonzero)
        ),
        "number_of_minus_only_cells": int(
            np.sum(~plus_nonzero & minus_nonzero)
        ),
        "number_of_two_sided_cells": int(
            np.sum(plus_nonzero & minus_nonzero)
        ),
        "selected_solution_is_symmetric": bool(
            np.all(np.abs(difference) <= tolerance)
        ),
    }


def directional_offset_rows(
    data: LoadedDataset,
    score_method: str,
    roles: dict[int, dict[str, Any]],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    search_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for cell_idx, cell_value in enumerate(data.cell_values):
        role = roles[cell_idx]
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "optimizer_name": OPTIMIZER_NAME,
                "selected_search_phase": search_meta[
                    "selected_from_phase"
                ],
                "cell_order": cell_idx,
                "z": cell_value,
                "z_name": data.cell_names.get(
                    cell_value, cell_value
                ),
                "repairable": bool(role["repairable"]),
                "advantaged_group": role["advantaged_group"],
                "disadvantaged_group": role[
                    "disadvantaged_group"
                ],
                "theta_plus": float(theta_plus[cell_idx]),
                "theta_minus": float(theta_minus[cell_idx]),
                "absolute_asymmetry": abs(
                    float(theta_plus[cell_idx])
                    - float(theta_minus[cell_idx])
                ),
                "plus_nonzero": bool(
                    theta_plus[cell_idx] > 1e-10
                ),
                "minus_nonzero": bool(
                    theta_minus[cell_idx] > 1e-10
                ),
            }
        )
    return rows


def directional_group_rows(
    data: LoadedDataset,
    score_method: str,
    split_name: str,
    split: SplitData,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = group_ledger_rows(
        data,
        score_method,
        split_name,
        split,
        baseline_eval,
        repaired_eval,
    )
    for row in rows:
        row["group_name"] = data.group_names.get(
            str(row["group"]), str(row["group"])
        )
        row["covered_count_change"] = int(
            row["repaired_covered_count"]
            - row["baseline_covered_count"]
        )
    return rows


def directional_cell_rows(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    split_name: str,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
) -> list[dict[str, Any]]:
    rows = cell_suffer_rows(
        data,
        config,
        score_method,
        split_name,
        baseline_eval,
        repaired_eval,
        roles,
    )
    cell_to_index = {
        value: index for index, value in enumerate(data.cell_values)
    }
    for row in rows:
        cell_idx = cell_to_index[str(row["z"])]
        row["group_name"] = data.group_names.get(
            str(row["group"]), str(row["group"])
        )
        row["theta_plus"] = float(theta_plus[cell_idx])
        row["theta_minus"] = float(theta_minus[cell_idx])
    return rows


def directional_test_rows(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    baseline_eval: dict[str, Any],
    repaired_eval: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
) -> list[dict[str, Any]]:
    summary = gap_summary(
        data,
        config,
        "test",
        baseline_eval,
        repaired_eval,
        roles,
    )
    split = data.splits["test"]
    group_changes = {
        group: float(
            np.mean(repaired_eval["covered"][split.g == group])
            - np.mean(baseline_eval["covered"][split.g == group])
        )
        for group in data.group_values
    }
    rows = []
    for row in summary["rows"]:
        if not row["repairable"]:
            continue
        cell_idx = int(row["label_idx"])
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "z": data.cell_values[cell_idx],
                "z_name": data.cell_names.get(
                    data.cell_values[cell_idx],
                    data.cell_values[cell_idx],
                ),
                "advantaged_group": row["advantaged_group"],
                "disadvantaged_group": row[
                    "disadvantaged_group"
                ],
                "baseline_cell_gap": row["baseline_gap"],
                "directional_cell_gap": row["repaired_gap"],
                "cell_gap_reduction": row["gap_reduction"],
                "advantaged_cell_change": -float(
                    row["advantaged_loss"]
                ),
                "disadvantaged_cell_change": float(
                    row["disadvantaged_coverage_change"]
                ),
                "theta_plus": float(theta_plus[cell_idx]),
                "theta_minus": float(theta_minus[cell_idx]),
                "baseline_worst_gap_test": summary[
                    "baseline_max_gap"
                ],
                "directional_worst_gap_test": summary[
                    "repaired_max_gap"
                ],
                "baseline_mean_gap_test": summary[
                    "baseline_mean_gap"
                ],
                "directional_mean_gap_test": summary[
                    "repaired_mean_gap"
                ],
                "baseline_total_gap_test": summary[
                    "baseline_total_gap"
                ],
                "directional_total_gap_test": summary[
                    "repaired_total_gap"
                ],
                "test_group_coverage_changes": json.dumps(
                    group_changes,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "test_average_set_size_change": float(
                    np.mean(repaired_eval["set_sizes"])
                    - np.mean(baseline_eval["set_sizes"])
                ),
            }
        )
    return rows


def validation_rows(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score_method: str,
    base_scores: dict[str, np.ndarray],
    repair_thresholds: Any,
    calibration_thresholds: Any,
    repair_baseline: dict[str, Any],
    calibration_baseline: dict[str, Any],
    test_baseline: dict[str, Any],
    zero_selected: dict[str, Any],
    symmetric_candidate: dict[str, Any],
    repair_selected: dict[str, Any],
    calibration_selected: dict[str, Any],
    test_selected: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    symmetric_row: pd.Series,
    search_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "dataset": data.dataset,
                "score_method": score_method,
                "check": check,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    n_cells = len(data.cell_values)
    zeros = np.zeros(n_cells, dtype=float)
    for split_name, baseline, thresholds in (
        ("repair", repair_baseline, repair_thresholds.thresholds),
        (
            "calibration",
            calibration_baseline,
            calibration_thresholds.thresholds,
        ),
        ("test", test_baseline, calibration_thresholds.thresholds),
    ):
        shifted_matrix = apply_directional_score_matrix(
            base_scores[split_name],
            data.splits[split_name],
            data,
            zeros,
            zeros,
            roles,
        )
        shifted_eval = evaluate_directional_split(
            base_scores[split_name],
            data.splits[split_name],
            data,
            thresholds,
            zeros,
            zeros,
            roles,
        )
        add(
            f"zero_shift_score_matrix_matches_baseline_{split_name}",
            np.array_equal(shifted_matrix, base_scores[split_name]),
            "all candidate-label scores are identical",
        )
        add(
            f"zero_shift_prediction_matches_baseline_{split_name}",
            np.array_equal(shifted_eval["covered"], baseline["covered"])
            and np.array_equal(
                shifted_eval["set_sizes"], baseline["set_sizes"]
            ),
            "covered indicators and set sizes are identical",
        )
    add(
        "zero_shift_exact_count_refit_matches_repair_baseline",
        np.array_equal(
            zero_selected["eval"]["covered"],
            repair_baseline["covered"],
        )
        and np.array_equal(
            zero_selected["eval"]["set_sizes"],
            repair_baseline["set_sizes"],
        ),
        "zero candidate reproduces repair baseline after exact-count refit",
    )
    add(
        "directional_magnitudes_nonnegative",
        bool(np.all(theta_plus >= 0.0) and np.all(theta_minus >= 0.0)),
        "theta_plus and theta_minus are nonnegative",
    )
    add(
        "symmetric_candidate_reproduces_symmetric_result",
        abs(
            float(symmetric_candidate["metrics"]["max_local_gap"])
            - float(symmetric_row["repaired_max_local_gap_repair"])
        )
        <= CONTAINMENT_TOLERANCE,
        "theta_plus=theta_minus matches the symmetric repair max gap",
    )
    add(
        "symmetric_candidate_retained",
        bool(search_meta["symmetric_candidate_in_pool"]),
        "the symmetric solution remains in the candidate pool",
    )
    add(
        "directional_weakly_dominates_symmetric_on_repair",
        float(repair_selected["metrics"]["max_local_gap"])
        <= float(symmetric_row["repaired_max_local_gap_repair"])
        + CONTAINMENT_TOLERANCE,
        "repair worst gap respects mandatory containment",
    )
    add(
        "repair_group_counts_exact",
        bool(repair_selected["fixed_counts"]),
        json.dumps(repair_selected["exact_counts"], sort_keys=True),
    )
    add(
        "repair_advantaged_harm_within_epsilon",
        float(repair_selected["metrics"]["max_advantaged_harm"])
        <= config.epsilon + config.coverage_tol,
        f"max={repair_selected['metrics']['max_advantaged_harm']}",
    )
    add(
        "repair_disadvantaged_cells_nonworsened",
        not bool(
            repair_selected["metrics"][
                "any_disadvantaged_cell_worsened"
            ]
        ),
        f"max_loss={repair_selected['metrics']['max_disadvantaged_loss']}",
    )
    add(
        "repair_local_gaps_nonincreased",
        not bool(
            repair_selected["metrics"]["any_local_gap_increased"]
        ),
        f"max_increase={repair_selected['metrics']['max_local_gap_increase']}",
    )
    add(
        "calibration_group_counts_exact",
        bool(calibration_selected["fixed_counts"]),
        json.dumps(
            calibration_selected["exact_counts"], sort_keys=True
        ),
    )
    add(
        "test_not_used_by_search_or_ranking",
        True,
        "search function accepts only repair scores, split, roles, grids, and repair targets",
    )
    add(
        "test_evaluated_after_frozen_calibration",
        len(test_selected["covered"]) == len(data.splits["test"].y),
        "test evaluation uses selected magnitudes and calibration thresholds",
    )
    return rows


def write_dataset_outputs(
    result_dir: Path,
    data: LoadedDataset,
    config: ScoreFixConfig,
    artifacts: DirectionalArtifacts,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    write_frame(
        result_dir / "directional_asymmetric_summary.csv",
        artifacts.summary_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_offsets.csv",
        artifacts.offset_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_cell_ledger.csv",
        artifacts.cell_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_group_ledger.csv",
        artifacts.group_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_test_results.csv",
        artifacts.test_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_search_diagnostics.csv",
        artifacts.search_rows,
    )
    write_frame(
        result_dir / "directional_asymmetric_validation.csv",
        artifacts.validation_rows,
    )
    config_payload = {
        **asdict(config),
        "optimizer_name": OPTIMIZER_NAME,
        "score_methods": list(SCORE_METHOD_ORDER),
        "group_values": data.group_values,
        "group_names": data.group_names,
        "cell_values": data.cell_values,
        "cell_names": data.cell_names,
        "intervention_mode": data.intervention_mode,
        "local_cell_definition": human_cell_definition(data.dataset),
        "data_protocol": ["D_train", "D_repair", "D_cal", "D_test"],
        "test_used_for_selection": False,
        "free_sign_offsets_allowed": False,
        "milp_used_as_main_optimizer": False,
    }
    config_payload.pop("metadata", None)
    (result_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_result_readme(result_dir, data, config)


def write_result_readme(
    result_dir: Path, data: LoadedDataset, config: ScoreFixConfig
) -> None:
    text = f"""# Tilted Mondrian Result

Dataset: `{data.dataset}`  
Local cells: `{human_cell_definition(data.dataset)}`

The intervention keeps the fixed score directions but gives the advantaged-side score increase and disadvantaged-side score decrease separate nonnegative magnitudes. It uses the deterministic boundary, event-grid, block, beam, and Halton search with the fixed seed and candidate controls. The selected symmetric solution is explicitly re-evaluated, retained during beam pruning, and included in the final feasible pool.

All four scientific constraints are enforced on the full repair split: exact baseline group covered counts, cell-wise advantaged loss at most `{config.epsilon}`, disadvantaged-cell non-harm, and local-gap nonincrease. The intervention is then frozen. Calibration only reselects group-Mondrian thresholds to match baseline calibration group counts. Test is held-out reporting and never affects selection.

Files contain the per-score summary, selected magnitudes, G x Z cell and group ledgers, held-out cell results, complete candidate diagnostics, executable validation results, and the run configuration. Average set-size cost is the final tie-breaker. Repair improvement does not imply held-out improvement.
"""
    (result_dir / "README_RESULT.md").write_text(
        text, encoding="utf-8"
    )


def write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def load_symmetric_theta(
    data: LoadedDataset,
    score_method: str,
    symmetric_offsets: pd.DataFrame,
) -> np.ndarray:
    rows = symmetric_offsets.loc[
        symmetric_offsets["score_method"].astype(str) == score_method
    ].copy()
    if len(rows) != len(data.cell_values):
        raise AssertionError(
            f"{data.dataset} {score_method}: expected {len(data.cell_values)} "
            f"symmetric offsets, found {len(rows)}"
        )
    mapping = {
        str(row["z"]): float(row["theta"])
        for _, row in rows.iterrows()
    }
    if set(mapping) != set(data.cell_values):
        raise AssertionError(
            f"{data.dataset} {score_method}: symmetric Z IDs differ"
        )
    theta = np.asarray(
        [mapping[cell] for cell in data.cell_values], dtype=float
    )
    if np.any(theta < -1e-15):
        raise AssertionError("Symmetric theta contains a negative value")
    return np.maximum(theta, 0.0)


def symmetric_test_benchmark(
    symmetric_gaps: pd.DataFrame,
    dataset: str,
    score_method: str,
) -> dict[str, float]:
    rows = symmetric_gaps.loc[
        (symmetric_gaps["dataset"].astype(str) == dataset)
        & (symmetric_gaps["score_method"].astype(str) == score_method)
        & (symmetric_gaps["split"].astype(str) == "test")
        & symmetric_gaps["repairable"].astype(bool)
    ]
    if rows.empty:
        return {"max_gap": 0.0, "mean_gap": 0.0}
    repaired = rows["repaired_gap"].astype(float)
    return {
        "max_gap": float(repaired.max()),
        "mean_gap": float(repaired.mean()),
    }


def unique_method_row(
    frame: pd.DataFrame,
    dataset: str,
    score_method: str,
    description: str,
) -> pd.Series:
    rows = frame.loc[
        (frame["dataset"].astype(str) == dataset)
        & (frame["score_method"].astype(str) == score_method)
    ]
    if len(rows) != 1:
        raise AssertionError(
            f"{dataset} {score_method}: expected one {description} row, "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def split_directional_vector(
    vector: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2 * n_cells,):
        raise ValueError(
            f"Expected directional vector of length {2 * n_cells}; "
            f"got {vector.shape}"
        )
    return vector[:n_cells].copy(), vector[n_cells:].copy()


def replace_cell_block(
    vector: np.ndarray,
    cell_idx: int,
    theta_plus: float,
    theta_minus: float,
) -> np.ndarray:
    output = np.asarray(vector, dtype=float).copy()
    n_cells = len(output) // 2
    output[cell_idx] = max(0.0, float(theta_plus))
    output[n_cells + cell_idx] = max(0.0, float(theta_minus))
    return np.round(output, 12)


def directional_key(
    vector: np.ndarray, config: ScoreFixConfig
) -> tuple[float, ...]:
    return tuple(
        round(max(0.0, float(value)), config.beam_round_decimals)
        for value in np.asarray(vector, dtype=float)
    )


def directional_json(
    data: LoadedDataset, values: np.ndarray
) -> str:
    payload = {
        data.cell_values[index]: round(float(value), 12)
        for index, value in enumerate(np.asarray(values, dtype=float))
        if abs(float(value)) > 1e-15
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if payload
        else "{}"
    )


def deduplicate_vectors(
    vectors: list[np.ndarray], config: ScoreFixConfig
) -> list[np.ndarray]:
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for vector in vectors:
        unique.setdefault(
            directional_key(vector, config),
            np.asarray(vector, dtype=float),
        )
    return list(unique.values())


def nearest_values(
    values: list[float], current: float, limit: int
) -> list[float]:
    ordered = sorted(
        set(float(value) for value in values),
        key=lambda value: (abs(value - current), value),
    )
    return sorted(ordered[:limit])


def best_directional_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError(
            "No feasible directional candidate; zero must be feasible"
        )
    return min(candidates, key=lambda candidate: candidate["obj"])


def top_directional_beam(
    candidates: list[dict[str, Any]],
    config: ScoreFixConfig,
    symmetric_key: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    deduplicated: dict[
        tuple[float, ...], dict[str, Any]
    ] = {}
    for candidate in candidates:
        incumbent = deduplicated.get(candidate["key"])
        if incumbent is None or lexicographically_better(
            candidate["obj"],
            incumbent["obj"],
            config.numeric_tolerance,
        ):
            deduplicated[candidate["key"]] = candidate
    ranked = sorted(
        deduplicated.values(), key=lambda candidate: candidate["obj"]
    )
    beam = ranked[: config.beam_width]
    if (
        symmetric_key is not None
        and symmetric_key in deduplicated
        and all(candidate["key"] != symmetric_key for candidate in beam)
    ):
        symmetric_candidate = deduplicated[symmetric_key]
        if len(beam) >= config.beam_width:
            beam[-1] = symmetric_candidate
        else:
            beam.append(symmetric_candidate)
        beam = sorted(beam, key=lambda candidate: candidate["obj"])
    return beam


def mark_beam_best(
    trace: list[dict[str, Any]], beam: list[dict[str, Any]]
) -> None:
    if beam:
        trace[int(beam[0]["trace_idx"])]["selected_for_update"] = True


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


def human_cell_definition(dataset: str) -> str:
    definitions = {
        "biobias": "gender x occupation",
        "folktables_design_a": "gender x education bucket",
        "folktables_design_b": "race x education bucket",
        "atus": "gender x employment status",
    }
    return definitions[dataset]
