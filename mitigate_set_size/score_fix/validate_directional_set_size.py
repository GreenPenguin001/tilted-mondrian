from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HERE))

from directional_set_size_core import (  # noqa: E402
    DirectionalSetSizeConfig,
    ProtocolEventTrace,
    apply_set_size_score_matrix,
    apply_set_size_score_matrix_fast,
    build_set_size_directional_grids,
    build_set_size_metric_context,
    compare_objective_and_key,
    define_set_size_roles,
    directional_key,
    directional_set_size_search,
    evaluate_set_size_split_direct,
    evaluate_set_size_with_fixed_counts,
    membership_event_values,
    set_size_candidate_metrics,
    set_size_config_from_score_config,
    set_size_objective,
    set_size_rejection_reasons,
    split_cell_statistics,
)
from input_loading import LoadedDataset, SplitData  # noqa: E402
from no_free_repair_core import (  # noqa: E402
    compute_baseline_group_thresholds,
    evaluate_split,
    labelwise_scores_from_probs,
)
from run_directional_set_size import load_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Tilted Mondrian set-size repair."
    )
    parser.add_argument(
        "--dataset", choices=("folktables_design_b", "biobias"), required=True
    )
    parser.add_argument("--score", choices=("simple",), required=True)
    parser.add_argument("--input-root", type=Path, default=HERE.parents[1] / "outputs")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional validation CSV destination under the package root.",
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--cross-cell-output", type=Path)
    parser.add_argument("--critical-event-output", type=Path)
    return parser.parse_args()


def check(name: str, passed: bool, detail: Any = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def synthetic_context(
    baseline_first: float,
    baseline_second: float,
    neutral: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    groups = ["0", "1"]
    roles = {
        0: {
            "supported": True,
            "baseline_order_tie": neutral,
            "pressure_active": not neutral,
            "larger_set_group": None if neutral else "0",
            "smaller_set_group": None if neutral else "1",
            "plus_group": None if neutral else "0",
            "minus_group": None if neutral else "1",
            "baseline_gap": abs(baseline_first - baseline_second),
        }
    }
    context = {
        "groups": groups,
        "cell_indices": {("0", 0): np.asarray([0]), ("1", 0): np.asarray([1])},
        "baseline_size": {("0", 0): baseline_first, ("1", 0): baseline_second},
        "supported_cells": [0],
        "active_cells": [] if neutral else [0],
        "baseline_avg_set_size": 0.5 * (baseline_first + baseline_second),
    }
    return roles, context


def synthetic_metrics(
    repaired_first: float,
    repaired_second: float,
    roles: dict[int, dict[str, Any]],
    context: dict[str, Any],
    config: DirectionalSetSizeConfig,
) -> dict[str, Any]:
    evaluation = {
        "set_sizes": np.asarray([repaired_first, repaired_second]),
        "covered": np.asarray([True, True]),
    }
    return set_size_candidate_metrics(
        evaluation, roles, np.zeros(2), context, config
    )


def run_input_validation(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repair = data.splits["repair"]
    scores = labelwise_scores_from_probs(
        repair.probs,
        score_method,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )
    thresholds = compute_baseline_group_thresholds(
        scores, repair, data.group_values, config.alpha
    )
    baseline = evaluate_split(scores, repair, thresholds.thresholds)
    roles = define_set_size_roles(repair, baseline, data, config)
    context = build_set_size_metric_context(scores, repair, baseline, data, roles)
    n_cells = len(data.cell_values)
    zero = np.zeros(n_cells)
    zero_trial = evaluate_set_size_with_fixed_counts(
        scores,
        repair,
        data,
        zero,
        zero,
        roles,
        thresholds.covered_counts,
        baseline,
        config,
        context,
        enforce_feasibility=True,
        retain_arrays=True,
    )
    zero_shifted = apply_set_size_score_matrix(
        scores, repair, data, zero, zero, roles
    )
    rows.extend(
        [
            check("theta_zero_scores_exact", np.array_equal(zero_shifted, scores)),
            check("theta_zero_set_sizes_exact", np.array_equal(zero_trial["eval"]["set_sizes"], baseline["set_sizes"])),
            check("theta_zero_covered_exact", np.array_equal(zero_trial["eval"]["covered"], baseline["covered"])),
            check("theta_zero_feasible", zero_trial["feasible"]),
            check("theta_zero_group_counts_exact", zero_trial["fixed_counts"]),
            check("every_supported_cell_checked", zero_trial["metrics"]["checked_supported_cell_count"] == len(context["supported_cells"])),
        ]
    )

    fast_scores = apply_set_size_score_matrix_fast(
        scores, data.intervention_mode, zero, zero, context
    )
    direct = evaluate_set_size_split_direct(
        scores, repair, data, thresholds.thresholds, zero, zero, roles
    )
    rows.append(check("fast_score_matrix_matches_direct", np.array_equal(fast_scores, zero_shifted)))
    rows.append(check("direct_set_sizes_match_production", np.array_equal(direct["set_sizes"], zero_trial["eval"]["set_sizes"])))

    neutral_roles, neutral_context = synthetic_context(1.0, 1.0, True)
    neutral_metrics = synthetic_metrics(1.0, 1.0, neutral_roles, neutral_context, config)
    neutral_good_reasons = set_size_rejection_reasons(True, neutral_metrics, config, neutral_roles)
    neutral_bad_metrics = synthetic_metrics(1.03, 1.03, neutral_roles, neutral_context, config)
    neutral_bad_reasons = set_size_rejection_reasons(True, neutral_bad_metrics, config, neutral_roles)
    rows.extend(
        [
            check("neutral_cell_has_no_direction", neutral_roles[0]["larger_set_group"] is None and neutral_roles[0]["smaller_set_group"] is None and not neutral_roles[0]["pressure_active"]),
            check("neutral_baseline_feasible", not neutral_good_reasons),
            check("neutral_budget_violation_rejected", "neutral_cell_violation" in neutral_bad_reasons),
        ]
    )

    nonneutral_roles, nonneutral_context = synthetic_context(2.0, 1.0, False)
    gap_metrics = synthetic_metrics(2.0, 0.8, nonneutral_roles, nonneutral_context, config)
    larger_metrics = synthetic_metrics(2.1, 1.1, nonneutral_roles, nonneutral_context, config)
    smaller_metrics = synthetic_metrics(2.0, 1.03, nonneutral_roles, nonneutral_context, config)
    rows.extend(
        [
            check("gap_increase_rejection_exercised", "set_size_gap_increased" in set_size_rejection_reasons(True, gap_metrics, config, nonneutral_roles)),
            check("larger_side_rejection_exercised", "larger_side_increased" in set_size_rejection_reasons(True, larger_metrics, config, nonneutral_roles)),
            check("smaller_side_rejection_exercised", "smaller_side_budget_exceeded" in set_size_rejection_reasons(True, smaller_metrics, config, nonneutral_roles)),
            check("group_count_rejection_exercised", "group_coverage_not_fixed" in set_size_rejection_reasons(False, neutral_metrics, config, neutral_roles)),
        ]
    )

    base_metrics = {
        "max_set_size_gap": 1.0,
        "mean_set_size_gap": 0.5,
        "total_set_size_gap_reduction": 0.0,
        "shift_norm": 0.0,
    }
    objective_better = dict(base_metrics, max_set_size_gap=0.9, shift_norm=10.0)
    objective_same = dict(base_metrics, shift_norm=1.0)
    base_obj = set_size_objective(base_metrics, base_metrics, config)
    better_obj = set_size_objective(objective_better, base_metrics, config)
    same_obj = set_size_objective(objective_same, base_metrics, config)
    rows.extend(
        [
            check("objective_prefers_worst_gap", compare_objective_and_key(better_obj, (1.0,), base_obj, (0.0,), config) < 0),
            check("objective_prefers_zero_norm_when_gaps_tie", compare_objective_and_key(base_obj, (0.0,), same_obj, (1.0,), config) < 0),
            check("candidate_key_breaks_complete_tie", compare_objective_and_key(base_obj, (0.0,), base_obj, (1.0,), config) < 0),
        ]
    )

    altered = dict(neutral_metrics)
    altered["fine_grained_coverage_gap"] = 999.0
    rows.append(
        check(
            "fine_grained_coverage_not_in_objective_or_feasibility",
            set_size_objective(altered, None, config) == set_size_objective(neutral_metrics, None, config)
            and set_size_rejection_reasons(True, altered, config, neutral_roles) == neutral_good_reasons,
        )
    )
    parameters = set(inspect.signature(directional_set_size_search).parameters)
    rows.append(
        check(
            "search_has_no_calibration_or_test_argument",
            "calibration" not in parameters and "test" not in parameters,
            sorted(parameters),
        )
    )

    protocol = ProtocolEventTrace()
    premature_test_rejected = False
    try:
        protocol.record("test_started", theta=np.zeros(2 * n_cells), thresholds={})
    except RuntimeError:
        premature_test_rejected = True
    rows.append(check("protocol_rejects_premature_test", premature_test_rejected))

    protocol = ProtocolEventTrace()
    protocol.record("repair_ready")
    protocol.record("search_started")
    protocol.record("theta_selected", theta=np.zeros(2 * n_cells))
    frozen_theta = np.zeros(2 * n_cells)
    protocol.record("theta_frozen", theta=frozen_theta)
    protocol.record("calibration_started", theta=frozen_theta)
    protocol.record("calibration_scores_constructed", theta=frozen_theta)
    protocol.record("calibration_baseline_evaluated", theta=frozen_theta)
    protocol.record(
        "calibration_thresholds_frozen",
        theta=frozen_theta,
        thresholds=thresholds.thresholds,
    )
    protocol.record(
        "test_started", theta=frozen_theta, thresholds=thresholds.thresholds
    )
    protocol.record(
        "test_scores_constructed",
        theta=frozen_theta,
        thresholds=thresholds.thresholds,
    )
    protocol.record(
        "baseline_test_evaluated",
        theta=frozen_theta,
        thresholds=thresholds.thresholds,
    )
    protocol.record(
        "repaired_test_evaluated",
        theta=frozen_theta,
        thresholds=thresholds.thresholds,
    )
    protocol.assert_complete()
    rows.append(check("protocol_state_machine_complete", all(protocol.validation().values())))

    grids = build_set_size_directional_grids(
        scores, repair, data, roles, thresholds.thresholds, config
    )
    active = [idx for idx, role in roles.items() if role["pressure_active"]]
    critical_ok = False
    critical_detail: dict[str, Any] = {}
    if active:
        idx = active[0]
        events, detail = membership_event_values(
            scores,
            repair,
            data,
            roles,
            idx,
            "plus",
            thresholds.thresholds,
            zero,
            zero,
            config,
        )
        expected_label_count = (
            scores.shape[1]
            if data.intervention_mode == "covariate_defined_z"
            else 1
        )
        off_diagonal_eligible = (
            data.intervention_mode != "label_defined_z"
            or any(int(value) != idx for value in detail["realized_y_values"])
        )
        critical_ok = bool(
            len(events)
            and detail["candidate_label_count"] == expected_label_count
            and off_diagonal_eligible
            and len(grids["plus"][idx]) == len(config.grid_multipliers)
        )
        critical_detail = detail
    rows.append(check("set_size_membership_critical_events", critical_ok, critical_detail))
    return rows


def biobias_cross_cell_diagnostics(
    data: LoadedDataset,
    config: DirectionalSetSizeConfig,
    score_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if data.intervention_mode != "label_defined_z":
        raise ValueError("BioBias diagnostics require label-defined Z")
    repair = data.splits["repair"]
    scores = labelwise_scores_from_probs(
        repair.probs,
        score_method,
        raps_lambda=config.raps_lambda,
        raps_k_reg=config.raps_k_reg,
        saps_lambda=config.saps_lambda,
    )
    thresholds = compute_baseline_group_thresholds(
        scores, repair, data.group_values, config.alpha
    )
    baseline = evaluate_split(scores, repair, thresholds.thresholds)
    roles = define_set_size_roles(repair, baseline, data, config)
    context = build_set_size_metric_context(scores, repair, baseline, data, roles)
    grids = build_set_size_directional_grids(
        scores, repair, data, roles, thresholds.thresholds, config
    )
    n_cells = len(data.cell_values)
    zero = np.zeros(n_cells, dtype=float)

    source = inspect.getsource(apply_set_size_score_matrix)

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    active_cells = [idx for idx, role in roles.items() if role["pressure_active"]]
    for cell_idx in active_cells:
        for side in ("plus", "minus"):
            events, event_detail = membership_event_values(
                scores,
                repair,
                data,
                roles,
                cell_idx,
                side,
                thresholds.thresholds,
                zero,
                zero,
                config,
            )
            positive_grid = [
                float(value) for value in grids[side][cell_idx] if value > 0.0
            ]
            diagnostic_values = []
            if positive_grid:
                diagnostic_values.extend(
                    [positive_grid[0], positive_grid[len(positive_grid) // 2], positive_grid[-1]]
                )
            if len(events):
                diagnostic_values.extend(
                    [float(np.quantile(events, 0.25)), float(np.quantile(events, 0.5))]
                )
            diagnostic_values = sorted(
                {round(value, 12) for value in diagnostic_values if value > 0.0}
            )
            for perturbation in diagnostic_values:
                plus = np.zeros(n_cells, dtype=float)
                minus = np.zeros(n_cells, dtype=float)
                (plus if side == "plus" else minus)[cell_idx] = perturbation
                trial = evaluate_set_size_with_fixed_counts(
                    scores,
                    repair,
                    data,
                    plus,
                    minus,
                    roles,
                    thresholds.covered_counts,
                    baseline,
                    config,
                    context,
                    enforce_feasibility=False,
                    retain_arrays=True,
                )
                direct = evaluate_set_size_split_direct(
                    scores,
                    repair,
                    data,
                    trial["thresholds"].thresholds,
                    plus,
                    minus,
                    roles,
                )
                production_stats = split_cell_statistics(
                    repair, trial["eval"], data
                )
                direct_stats = split_cell_statistics(repair, direct, data)
                baseline_stats = split_cell_statistics(repair, baseline, data)
                max_discrepancy = float(
                    np.max(
                        np.abs(
                            np.asarray(trial["eval"]["set_sizes"])
                            - np.asarray(direct["set_sizes"])
                        )
                    )
                )
                stat_discrepancies = []
                off_diagonal = []
                rows: list[dict[str, Any]] = []
                for realized_idx, cell_value in enumerate(data.cell_values):
                    for group in data.group_values:
                        production_mean = float(
                            production_stats[(group, realized_idx)]["mean_set_size"]
                        )
                        direct_mean = float(
                            direct_stats[(group, realized_idx)]["mean_set_size"]
                        )
                        baseline_mean = float(
                            baseline_stats[(group, realized_idx)]["mean_set_size"]
                        )
                        delta = production_mean - baseline_mean
                        stat_discrepancies.append(abs(production_mean - direct_mean))
                        is_off_diagonal = realized_idx != cell_idx
                        if is_off_diagonal and abs(delta) > config.set_size_tolerance:
                            off_diagonal.append((group, realized_idx, delta))
                        rows.append(
                            {
                                "dataset": data.dataset,
                                "score_method": score_method,
                                "candidate_label_idx": cell_idx,
                                "candidate_label": data.cell_values[cell_idx],
                                "candidate_label_name": data.cell_names[data.cell_values[cell_idx]],
                                "shift_side": side,
                                "shift_group": roles[cell_idx][f"{side}_group"],
                                "perturbation": perturbation,
                                "realized_y_idx": realized_idx,
                                "realized_y": cell_value,
                                "realized_y_name": data.cell_names[cell_value],
                                "group": group,
                                "baseline_mean_set_size": baseline_mean,
                                "production_mean_set_size": production_mean,
                                "direct_mean_set_size": direct_mean,
                                "delta_s": delta,
                                "is_off_diagonal": is_off_diagonal,
                            }
                        )
                max_stat_discrepancy = float(max(stat_discrepancies, default=0.0))
                attempt = {
                    "cell_idx": cell_idx,
                    "side": side,
                    "perturbation": perturbation,
                    "event_detail": event_detail,
                    "max_row_discrepancy": max_discrepancy,
                    "max_cell_mean_discrepancy": max_stat_discrepancy,
                    "off_diagonal_effect_count": len(off_diagonal),
                    "fixed_counts": bool(trial["fixed_counts"]),
                    "rows": rows,
                }
                attempts.append(attempt)
                if (
                    trial["fixed_counts"]
                    and max_discrepancy <= config.set_size_tolerance
                    and max_stat_discrepancy <= config.set_size_tolerance
                    and off_diagonal
                ):
                    selected = attempt
                    break
            if selected is not None:
                break
        if selected is not None:
            break

    if selected is None:
        raise AssertionError(
            "No deterministic BioBias perturbation exercised an off-diagonal effect"
        )
    cell_idx = int(selected["cell_idx"])
    side = str(selected["side"])
    event_detail = selected["event_detail"]
    probe_plus = np.zeros(n_cells, dtype=float)
    probe_minus = np.zeros(n_cells, dtype=float)
    (probe_plus if side == "plus" else probe_minus)[cell_idx] = float(
        selected["perturbation"]
    )
    altered_split = SplitData(
        probs=repair.probs,
        y=np.roll(np.asarray(repair.y, dtype=int), 1),
        g=repair.g,
        z=repair.z,
    )
    shifted_original = apply_set_size_score_matrix(
        scores, repair, data, probe_plus, probe_minus, roles
    )
    shifted_altered_y = apply_set_size_score_matrix(
        scores, altered_split, data, probe_plus, probe_minus, roles
    )
    leakage_free = bool(
        ".y" not in source
        and np.array_equal(shifted_original, shifted_altered_y)
    )
    eligible_realized = [int(value) for value in event_detail["realized_y_values"]]
    group = roles[cell_idx][f"{side}_group"]
    row_mask = repair.g == group
    affected = apply_set_size_score_matrix(
        scores, repair, data, zero, zero, roles
    )[row_mask, cell_idx]
    threshold = float(thresholds.thresholds[group])
    margins = threshold - affected if side == "plus" else affected - threshold
    contributing_realized = sorted(
        {
            int(value)
            for value in np.asarray(repair.y)[row_mask][
                np.isfinite(margins) & (margins > config.numeric_tolerance)
            ].tolist()
        }
    )
    off_diagonal_contributing = [
        value for value in contributing_realized if value != cell_idx
    ]
    critical_rows = [
        {
            "dataset": data.dataset,
            "score_method": score_method,
            "candidate_label_idx": cell_idx,
            "candidate_label": data.cell_values[cell_idx],
            "candidate_label_name": data.cell_names[data.cell_values[cell_idx]],
            "side": side,
            "shift_group": roles[cell_idx][f"{side}_group"],
            "event_count": event_detail["event_count"],
            "eligible_row_count": event_detail["row_count"],
            "candidate_labels": json.dumps(event_detail["candidate_labels"]),
            "eligible_realized_y_indices": json.dumps(eligible_realized),
            "contributing_realized_y_indices": json.dumps(contributing_realized),
            "off_diagonal_contributing_realized_y_indices": json.dumps(
                off_diagonal_contributing
            ),
            "grid_multiplier_count": len(config.grid_multipliers),
            "grid_value_count": len(grids[side][cell_idx]),
            "global_lambda_count": len(config.global_lambda_grid),
            "beam_width": config.beam_width,
            "quasirandom_candidate_count": config.quasirandom_candidate_count,
            "quasirandom_seed": config.quasirandom_seed,
        }
    ]
    validation_rows = [
        check("label_defined_shift_does_not_read_realized_y", leakage_free),
        check(
            "cross_cell_full_matrix_matches_production",
            selected["max_row_discrepancy"] <= config.set_size_tolerance
            and selected["max_cell_mean_discrepancy"] <= config.set_size_tolerance,
            max(
                selected["max_row_discrepancy"],
                selected["max_cell_mean_discrepancy"],
            ),
        ),
        check(
            "off_diagonal_realized_y_effect_exercised",
            selected["off_diagonal_effect_count"] > 0,
            selected["off_diagonal_effect_count"],
        ),
        check(
            "critical_events_include_off_diagonal_realized_y",
            bool(off_diagonal_contributing),
            off_diagonal_contributing,
        ),
        check(
            "diagnostic_exact_group_counts",
            selected["fixed_counts"],
        ),
    ]
    for row in validation_rows:
        row.setdefault("dataset", data.dataset)
        row.setdefault("score_method", score_method)
    return validation_rows, selected["rows"], critical_rows


def compare_result_dirs(first: Path, second: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deterministic_files = [
        "directional_set_size_summary.csv",
        "directional_set_size_offsets.csv",
        "directional_set_size_cell_ledger.csv",
        "directional_set_size_group_ledger.csv",
        "directional_set_size_search_diagnostics.csv",
        "directional_set_size_validation.csv",
        "directional_set_size_test_results.csv",
        "directional_set_size_protocol_events.csv",
    ]
    for name in deterministic_files:
        left = pd.read_csv(first / name)
        right = pd.read_csv(second / name)
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
            passed = True
            detail = "exact"
        except AssertionError as exc:
            passed = False
            detail = str(exc).splitlines()[0]
        rows.append(check(f"deterministic_{name}", passed, detail))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    root = HERE.parents[1].resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Validation output must remain under the package root {root}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(resolved, index=False)


def main() -> None:
    args = parse_args()
    from run_directional_set_size import default_input_dir

    input_dir = (args.input_dir or default_input_dir(args.dataset, args.input_root)).resolve()
    data, score_config = load_experiment(args.dataset, input_dir)
    config = set_size_config_from_score_config(score_config)
    input_validation_rows = run_input_validation(data, config, args.score)
    for row in input_validation_rows:
        row.setdefault("dataset", args.dataset)
        row.setdefault("score_method", args.score)
    rows: list[dict[str, Any]] = []
    if args.dataset == "biobias":
        diagnostic_rows, cross_cell_rows, critical_rows = (
            biobias_cross_cell_diagnostics(data, config, args.score)
        )
        rows.extend(diagnostic_rows)
        if args.cross_cell_output:
            write_rows(args.cross_cell_output, cross_cell_rows)
        if args.critical_event_output:
            write_rows(args.critical_event_output, critical_rows)
    if args.result_dir:
        runtime_path = args.result_dir / "directional_set_size_validation.csv"
        if runtime_path.exists():
            rows.extend(pd.read_csv(runtime_path).to_dict(orient="records"))
    rows.extend(input_validation_rows)
    if args.result_dir and args.comparison_dir:
        comparison_rows = compare_result_dirs(args.result_dir, args.comparison_dir)
        for row in comparison_rows:
            row.setdefault("dataset", args.dataset)
            row.setdefault("score_method", args.score)
        rows.extend(comparison_rows)
    if args.output:
        write_rows(args.output, rows)
    failed = [row for row in rows if not row["passed"]]
    for row in rows:
        print(f"{row['check']}: {'PASS' if row['passed'] else 'FAIL'}")
    if failed:
        raise SystemExit(f"Validation failed: {[row['check'] for row in failed]}")


if __name__ == "__main__":
    main()
