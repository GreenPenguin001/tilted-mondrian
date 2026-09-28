"""Run one split-specific symmetric and directional Table 2 configuration."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .source_registry import SCORES


COMMON = Path(__file__).resolve().parents[2] / "score_fix" / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from directional_asymmetric_core import (  # noqa: E402
    apply_directional_score_matrix,
    evaluate_directional_with_fixed_counts,
    run_directional_score_method,
)
from input_loading import LoadedDataset  # noqa: E402
from no_free_repair_core import (  # noqa: E402
    ScoreFixConfig,
    compute_baseline_group_thresholds,
    define_repair_roles,
    evaluate_split,
    labelwise_scores_from_probs,
    run_one_score_method,
)


REPORTING_SPLITS = {
    "baseline_max_gap_reporting_split": "test",
    "directional_max_gap_reporting_split": "test",
    "delta_max_gap_reporting_split": "test",
    "maximum_cell_mitigation_reporting_split": "test",
    "harm_reporting_split": "repair",
    "average_set_size_change_reporting_split": "test",
    "certificate_reporting_split": "repair",
}


def score_config(dataset: str, score: str) -> ScoreFixConfig:
    if score not in SCORES:
        raise ValueError(f"Unsupported score: {score}")
    return ScoreFixConfig(
        dataset=dataset,
        score_methods=(score,),
        quasirandom_seed=2026,
        raps_lambda=0.01,
        raps_k_reg=5,
        saps_lambda=0.2,
        metadata={"fixed_model": True, "optimizer_seed": 2026},
    )


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def selected_vector(offsets: pd.DataFrame, data: LoadedDataset, column: str) -> np.ndarray:
    mapping = {str(row["z"]): float(row[column]) for _, row in offsets.iterrows()}
    return np.asarray([mapping[cell] for cell in data.cell_values], dtype=float)


def compute_threshold_audit(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score: str,
    offsets: pd.DataFrame,
) -> dict[str, Any]:
    scores = {
        split: labelwise_scores_from_probs(
            split_data.probs,
            score,
            raps_lambda=config.raps_lambda,
            raps_k_reg=config.raps_k_reg,
            saps_lambda=config.saps_lambda,
        )
        for split, split_data in data.splits.items()
    }
    repair = data.splits["repair"]
    repair_thresholds = compute_baseline_group_thresholds(
        scores["repair"], repair, data.group_values, config.alpha
    )
    repair_baseline = evaluate_split(scores["repair"], repair, repair_thresholds.thresholds)
    roles = define_repair_roles(repair_baseline, data, config)
    theta_plus = selected_vector(offsets, data, "theta_plus")
    theta_minus = selected_vector(offsets, data, "theta_minus")
    shifted_repair = evaluate_directional_with_fixed_counts(
        base_scores=scores["repair"],
        split=repair,
        data=data,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        roles=roles,
        target_counts=repair_thresholds.covered_counts,
        baseline_eval=repair_baseline,
        config=config,
    )
    calibration = data.splits["calibration"]
    calibration_thresholds = compute_baseline_group_thresholds(
        scores["calibration"], calibration, data.group_values, config.alpha
    )
    calibration_baseline = evaluate_split(
        scores["calibration"], calibration, calibration_thresholds.thresholds
    )
    shifted_calibration = evaluate_directional_with_fixed_counts(
        base_scores=scores["calibration"],
        split=calibration,
        data=data,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        roles=roles,
        target_counts=calibration_thresholds.covered_counts,
        baseline_eval=calibration_baseline,
        config=config,
    )
    return {
        "baseline_repair_thresholds": repair_thresholds.thresholds,
        "baseline_repair_covered_counts": repair_thresholds.covered_counts,
        "shifted_repair_thresholds": shifted_repair["thresholds"].thresholds,
        "shifted_repair_covered_counts": shifted_repair["thresholds"].covered_counts,
        "baseline_calibration_thresholds": calibration_thresholds.thresholds,
        "baseline_calibration_covered_counts": calibration_thresholds.covered_counts,
        "shifted_calibration_thresholds": shifted_calibration["thresholds"].thresholds,
        "shifted_calibration_covered_counts": shifted_calibration["thresholds"].covered_counts,
        "repair_group_covered_counts_exact": bool(shifted_repair["fixed_counts"]),
        "calibration_group_covered_counts_exact": bool(shifted_calibration["fixed_counts"]),
        "repair_roles": {
            data.cell_values[index]: {
                "repairable": bool(role["repairable"]),
                "advantaged_group": role["advantaged_group"],
                "disadvantaged_group": role["disadvantaged_group"],
                "baseline_gap": float(role["baseline_gap"]),
            }
            for index, role in roles.items()
        },
    }


def repair_certificate(
    data: LoadedDataset,
    config: ScoreFixConfig,
    cell_ledger: pd.DataFrame,
) -> dict[str, Any]:
    repair = cell_ledger.loc[cell_ledger["split"].astype(str) == "repair"].copy()
    repairable_cells = []
    for cell in data.cell_values:
        rows = repair.loc[repair["z"].astype(str) == cell]
        if rows.empty or set(rows["role"].astype(str)).isdisjoint({"advantaged", "disadvantaged"}):
            continue
        repairable_cells.append(cell)
    group_totals = {
        str(group): float(repair.loc[repair["group"].astype(str) == str(group), "n"].sum())
        for group in data.group_values
    }
    bounds: list[dict[str, Any]] = []
    roles: dict[str, tuple[str, str]] = {}
    for cell in repairable_cells:
        rows = repair.loc[repair["z"].astype(str) == cell]
        adv = str(rows.loc[rows["role"] == "advantaged", "group"].iloc[0])
        dis = str(rows.loc[rows["role"] == "disadvantaged", "group"].iloc[0])
        roles[cell] = (adv, dis)
    for cell in repairable_cells:
        rows = repair.loc[repair["z"].astype(str) == cell]
        adv, dis = roles[cell]
        dis_row = rows.loc[rows["group"].astype(str) == dis].iloc[0]
        pi_dis = float(dis_row["n"]) / group_totals[dis]
        numerator = sum(
            float(
                repair.loc[
                    (repair["z"].astype(str) == other)
                    & (repair["group"].astype(str) == dis),
                    "n",
                ].iloc[0]
            )
            / group_totals[dis]
            for other, (other_adv, _) in roles.items()
            if other_adv == dis
        )
        coverages = rows.set_index(rows["group"].astype(str))["baseline_coverage"].astype(float)
        baseline_gap = abs(float(coverages.iloc[0]) - float(coverages.iloc[1]))
        budget_term = config.epsilon * (1.0 + numerator / pi_dis)
        bounds.append(
            {
                "z": cell,
                "z_name": data.cell_names.get(cell, cell),
                "bound": min(baseline_gap, budget_term),
                "active_term": "baseline_gap" if baseline_gap <= budget_term else "epsilon_budget",
            }
        )
    if not bounds:
        return {
            "repair_certificate": 0.0,
            "certificate_active_term": "none",
            "certificate_attaining_cell": "",
        }
    active = max(bounds, key=lambda item: float(item["bound"]))
    return {
        "repair_certificate": float(active["bound"]),
        "certificate_active_term": active["active_term"],
        "certificate_attaining_cell": active["z_name"],
    }


def reporting_source_checks(
    summary: dict[str, Any],
    cell_ledger: pd.DataFrame,
    group_ledger: pd.DataFrame,
    test_results: pd.DataFrame,
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    test_rows = test_results.copy()
    repair_cells = cell_ledger.loc[
        (cell_ledger["split"].astype(str) == "repair")
        & (cell_ledger["role"].astype(str) == "advantaged")
    ]
    test_groups = group_ledger.loc[group_ledger["split"].astype(str) == "test"]
    n_test = float(test_groups["n"].sum())
    weighted_size_change = float(
        (test_groups["set_size_change"].astype(float) * test_groups["n"].astype(float)).sum()
        / n_test
    )
    expected = {
        "baseline_test_max_gap": float(test_rows["baseline_cell_gap"].max()),
        "directional_test_max_gap": float(test_rows["directional_cell_gap"].max()),
        "directional_test_delta_max_gap": float(test_rows["baseline_cell_gap"].max() - test_rows["directional_cell_gap"].max()),
        "heldout_maximum_single_cell_mitigation": float(test_rows["cell_gap_reduction"].max()),
        "maximum_advantaged_harm_repair": float(
            max(0.0, (-repair_cells["coverage_change"].astype(float)).max())
        ),
        "test_average_set_size_change": weighted_size_change,
    }
    rows: list[dict[str, Any]] = []
    for field, value in expected.items():
        rows.append(
            {
                "check": f"ledger_reconstructs_{field}",
                "passed": abs(float(summary[field]) - value) <= tolerance,
                "reported": summary[field],
                "reconstructed": value,
            }
        )
    argmax = test_rows.loc[test_rows["cell_gap_reduction"].astype(float).idxmax(), "z_name"]
    rows.append(
        {
            "check": "ledger_reconstructs_heldout_maximum_mitigation_cell",
            "passed": str(summary["heldout_maximum_mitigation_cell"]) == str(argmax),
            "reported": summary["heldout_maximum_mitigation_cell"],
            "reconstructed": argmax,
        }
    )
    for field, split in REPORTING_SPLITS.items():
        rows.append(
            {
                "check": f"reporting_split_{field}",
                "passed": summary[field] == split,
                "reported": summary[field],
                "reconstructed": split,
            }
        )
    repair_max = float(summary["repair_maximum_single_cell_mitigation"])
    test_max = float(summary["heldout_maximum_single_cell_mitigation"])
    rows.append(
        {
            "check": "heldout_maximum_cell_not_substituted_from_repair",
            "passed": not (
                abs(test_max - repair_max) <= tolerance
                and abs(test_max - float(test_rows["cell_gap_reduction"].max())) > tolerance
            ),
            "reported": test_max,
            "reconstructed": float(test_rows["cell_gap_reduction"].max()),
        }
    )
    return rows


def build_long_summary(
    data: LoadedDataset,
    config: ScoreFixConfig,
    score: str,
    split_seed: int | str,
    symmetric: dict[str, list[dict[str, Any]]],
    directional: dict[str, list[dict[str, Any]]],
    threshold_audit: dict[str, Any],
    symmetric_runtime: float,
    directional_runtime: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directional_summary = dict(directional["summary"][0])
    symmetric_summary = dict(symmetric["summary"][0])
    test_results = frame(directional["test"])
    cell_ledger = frame(directional["cell"])
    group_ledger = frame(directional["group"])
    best_test = test_results.loc[test_results["cell_gap_reduction"].astype(float).idxmax()]
    delta = float(
        directional_summary["baseline_test_max_gap"]
        - directional_summary["directional_test_max_gap"]
    )
    if delta > config.numeric_tolerance:
        category = "positive"
    elif delta < -config.numeric_tolerance:
        category = "negative"
    else:
        category = "tie"
    validation = frame(directional["validation"])
    constraint_pass = bool(validation["passed"].astype(bool).all())
    theta = frame(directional["offset"])
    summary: dict[str, Any] = {
        "dataset": data.dataset,
        "score": score,
        "split_seed": split_seed,
        "optimizer_seed": 2026,
        "alpha": config.alpha,
        "epsilon": config.epsilon,
        "raps_lambda": config.raps_lambda,
        "raps_k_reg": config.raps_k_reg,
        "saps_lambda": config.saps_lambda,
        "min_cell_count": config.min_cell_count,
        "n_repair": len(data.splits["repair"].y),
        "n_calibration": len(data.splits["calibration"].y),
        "n_test": len(data.splits["test"].y),
        "repairable_cell_count": int(
            sum(bool(value["repairable"]) for value in threshold_audit["repair_roles"].values())
        ),
        "unsupported_cell_count": int(
            sum(not bool(value["repairable"]) for value in threshold_audit["repair_roles"].values())
        ),
        "repair_defined_roles": json.dumps(threshold_audit["repair_roles"], sort_keys=True),
        "baseline_repair_thresholds": json.dumps(threshold_audit["baseline_repair_thresholds"], sort_keys=True),
        "baseline_repair_covered_counts": json.dumps(threshold_audit["baseline_repair_covered_counts"], sort_keys=True),
        "symmetric_selected_theta": json.dumps(
            {str(row["z"]): float(row["theta"]) for row in symmetric["theta"]}, sort_keys=True
        ),
        "directional_theta_plus": json.dumps(
            {str(row["z"]): float(row["theta_plus"]) for row in directional["offset"]}, sort_keys=True
        ),
        "directional_theta_minus": json.dumps(
            {str(row["z"]): float(row["theta_minus"]) for row in directional["offset"]}, sort_keys=True
        ),
        "selected_search_phase": directional_summary["selected_search_phase"],
        "total_candidate_count": int(directional_summary["candidate_count_total"]),
        "feasible_candidate_count": int(directional_summary["candidate_count_feasible"]),
        "symmetric_candidate_flag": bool(directional_summary["symmetric_candidate_in_pool"]),
        "symmetric_candidate_equivalence_flag": bool(
            validation.loc[
                validation["check"] == "symmetric_candidate_reproduces_symmetric_result",
                "passed",
            ].all()
        ),
        "directional_repair_not_worse_than_symmetric_flag": float(directional_summary["directional_repair_max_gap"])
        <= float(directional_summary["symmetric_repair_max_gap"]) + 1e-8,
        "zero_intervention_selected_flag": bool(
            np.all(theta["theta_plus"].astype(float).to_numpy() <= config.numeric_tolerance)
            and np.all(theta["theta_minus"].astype(float).to_numpy() <= config.numeric_tolerance)
        ),
        "baseline_repair_max_gap": directional_summary["baseline_repair_max_gap"],
        "symmetric_repair_max_gap": directional_summary["symmetric_repair_max_gap"],
        "directional_repair_max_gap": directional_summary["directional_repair_max_gap"],
        "baseline_repair_mean_gap": directional_summary["baseline_repair_mean_gap"],
        "directional_repair_mean_gap": directional_summary["directional_repair_mean_gap"],
        "repair_maximum_single_cell_mitigation": directional_summary["repair_maximum_single_cell_mitigation"],
        "repair_total_mitigation": directional_summary["repair_total_mitigation"],
        "maximum_advantaged_harm_repair": directional_summary["maximum_advantaged_harm_repair"],
        "minimum_disadvantaged_cell_change_repair": directional_summary["minimum_disadvantaged_cell_change_repair"],
        "maximum_local_gap_increase_repair": directional_summary["maximum_local_gap_increase_repair"],
        "repair_group_covered_counts_exact": bool(directional_summary["repair_group_covered_counts_exact"]),
        "every_repair_constraint_pass": constraint_pass,
        "baseline_calibration_thresholds": json.dumps(threshold_audit["baseline_calibration_thresholds"], sort_keys=True),
        "shifted_calibration_thresholds": json.dumps(threshold_audit["shifted_calibration_thresholds"], sort_keys=True),
        "baseline_calibration_covered_counts": json.dumps(threshold_audit["baseline_calibration_covered_counts"], sort_keys=True),
        "shifted_calibration_covered_counts": json.dumps(threshold_audit["shifted_calibration_covered_counts"], sort_keys=True),
        "calibration_group_covered_counts_exact": bool(threshold_audit["calibration_group_covered_counts_exact"]),
        "baseline_test_max_gap": directional_summary["baseline_test_max_gap"],
        "symmetric_test_max_gap": directional_summary["symmetric_test_max_gap"],
        "directional_test_max_gap": directional_summary["directional_test_max_gap"],
        "directional_test_delta_max_gap": delta,
        "symmetric_test_delta_max_gap": float(
            directional_summary["baseline_test_max_gap"] - directional_summary["symmetric_test_max_gap"]
        ),
        "baseline_test_mean_gap": directional_summary["baseline_test_mean_gap"],
        "directional_test_mean_gap": directional_summary["directional_test_mean_gap"],
        "baseline_test_total_gap": directional_summary["baseline_test_total_gap"],
        "directional_test_total_gap": directional_summary["directional_test_total_gap"],
        "heldout_maximum_single_cell_mitigation": float(best_test["cell_gap_reduction"]),
        "heldout_maximum_mitigation_cell": str(best_test["z_name"]),
        "heldout_total_mitigation": directional_summary["directional_total_mitigation_test"],
        "test_average_set_size_change": directional_summary["test_average_set_size_change"],
        "heldout_group_coverage_changes": str(best_test["test_group_coverage_changes"]),
        "heldout_improvement_category": category,
        "baseline_max_gap_reporting_split": "test",
        "directional_max_gap_reporting_split": "test",
        "delta_max_gap_reporting_split": "test",
        "maximum_cell_mitigation_reporting_split": "test",
        "harm_reporting_split": "repair",
        "average_set_size_change_reporting_split": "test",
        "certificate_reporting_split": "repair",
        "symmetric_runtime_seconds": symmetric_runtime,
        "directional_runtime_seconds": directional_runtime,
        "total_runtime_seconds": symmetric_runtime + directional_runtime,
        "warnings": "",
        "failure_reason": "",
        **repair_certificate(data, config, cell_ledger),
    }
    checks = reporting_source_checks(summary, cell_ledger, group_ledger, test_results)
    if not all(bool(row["passed"]) for row in checks):
        raise AssertionError(f"Reporting-source validation failed: {checks}")
    return summary, checks


def write_frames(base: Path, mapping: dict[str, list[dict[str, Any]]], names: dict[str, str]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for key, filename in names.items():
        frame(mapping[key]).to_csv(base / filename, index=False)


def run_configuration(
    data: LoadedDataset,
    dataset: str,
    score: str,
    split_seed: int | str,
    output_dir: Path,
) -> dict[str, Any]:
    config = score_config(dataset, score)
    symmetric_start = time.perf_counter()
    symmetric = run_one_score_method(data, config, score)
    symmetric_runtime = float(time.perf_counter() - symmetric_start)
    directional_start = time.perf_counter()
    directional = run_directional_score_method(
        data=data,
        config=config,
        score_method=score,
        symmetric_summary=frame(symmetric["summary"]),
        symmetric_offsets=frame(symmetric["theta"]),
        symmetric_gaps=frame(symmetric["label"]),
    )
    directional_runtime = float(time.perf_counter() - directional_start)
    offsets = frame(directional["offset"])
    threshold_audit = compute_threshold_audit(data, config, score, offsets)
    summary, checks = build_long_summary(
        data=data,
        config=config,
        score=score,
        split_seed=split_seed,
        symmetric=symmetric,
        directional=directional,
        threshold_audit=threshold_audit,
        symmetric_runtime=symmetric_runtime,
        directional_runtime=directional_runtime,
    )
    write_frames(
        output_dir / "symmetric",
        symmetric,
        {
            "summary": "summary.csv",
            "group": "group_coverage_ledger.csv",
            "label": "label_gap_ledger.csv",
            "cell": "cell_suffer_ledger.csv",
            "trace": "optimizer_trace.csv",
            "theta": "selected_theta.csv",
        },
    )
    write_frames(
        output_dir / "directional",
        directional,
        {
            "summary": "directional_asymmetric_summary.csv",
            "offset": "directional_asymmetric_offsets.csv",
            "cell": "directional_asymmetric_cell_ledger.csv",
            "group": "directional_asymmetric_group_ledger.csv",
            "test": "directional_asymmetric_test_results.csv",
            "search": "directional_asymmetric_search_diagnostics.csv",
            "comparison": "comparison.csv",
            "validation": "directional_asymmetric_validation.csv",
        },
    )
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(validation_dir / "reporting_source_checks.csv", index=False)
    (validation_dir / "threshold_audit.json").write_text(
        json.dumps(threshold_audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    return summary
