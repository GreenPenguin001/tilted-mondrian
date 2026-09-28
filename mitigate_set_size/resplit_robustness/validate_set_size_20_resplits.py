from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent

sys.path.insert(0, str(HERE))
import run_set_size_20_resplits as robustness_runner  # noqa: E402
from run_set_size_20_resplits import (  # noqa: E402
    CORE_OUTPUTS,
    DATASETS,
    EXPECTED_EVENTS,
    RESPLIT_SEEDS,
    SCORES,
    as_bool,
    build_loaded_dataset,
    load_partition,
    load_pool,
    problem_directory,
)

RESULT_ROOT = robustness_runner.DEFAULT_RESULT_ROOT
REPORT_PATH = RESULT_ROOT / "set_size_20_resplit_robustness_report.md"


def configure_paths(protocol_root: Path, output_root: Path) -> None:
    global RESULT_ROOT, REPORT_PATH

    robustness_runner.configure_paths(protocol_root, output_root)
    RESULT_ROOT = robustness_runner.RESULT_ROOT
    REPORT_PATH = RESULT_ROOT / "set_size_20_resplit_robustness_report.md"

COMMON = EXPERIMENT_ROOT / "score_fix" / "common"
sys.path.insert(0, str(COMMON))
from no_free_repair_core import labelwise_scores_from_probs  # noqa: E402


DISPLAY_NAMES = {
    "biobias": "Bias in Bios",
    "folktables_design_a": "Folktables Design A",
    "folktables_design_b": "Folktables Design B",
    "atus": "ATUS",
}
TOLERANCE = 1e-9
SCIENTIFIC_CSV_FILES = (
    "directional_set_size_summary.csv",
    "directional_set_size_offsets.csv",
    "directional_set_size_cell_ledger.csv",
    "directional_set_size_group_ledger.csv",
    "directional_set_size_search_diagnostics.csv",
    "directional_set_size_validation.csv",
    "directional_set_size_test_results.csv",
    "directional_set_size_protocol_events.csv",
)
ABSOLUTE_METRICS = (
    "repair_worst_gap_abs_reduction",
    "repair_mean_gap_abs_reduction",
    "test_worst_gap_abs_reduction",
    "test_mean_gap_abs_reduction",
    "delta_overall_mean_set_size",
    "delta_overall_coverage",
    "delta_worst_coverage_gap",
    "test_max_abs_cell_coverage_change",
)
RELATIVE_METRICS = (
    "repair_worst_gap_rel_reduction_pct",
    "repair_mean_gap_rel_reduction_pct",
    "test_worst_gap_rel_reduction_pct",
    "test_mean_gap_rel_reduction_pct",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def relative_percent(reduction: float, baseline: float) -> float:
    return 100.0 * reduction / baseline if baseline > TOLERANCE else float("nan")


def classify_reduction(value: float) -> str:
    if value > TOLERANCE:
        return "improved"
    if value < -TOLERANCE:
        return "worsened"
    return "tied"


def classify_cells(cells: pd.DataFrame, split: str) -> tuple[int, int, int]:
    rows = (
        cells[(cells["split"] == split) & cells["supported"].map(as_bool)]
        .sort_values(["cell_idx", "group"])
        .drop_duplicates("cell_idx")
    )
    values = rows["set_size_gap_reduction"].to_numpy(float)
    improved = int(np.sum(values > TOLERANCE))
    worsened = int(np.sum(values < -TOLERANCE))
    return improved, int(len(values) - improved - worsened), worsened


def weighted_coverage(groups: pd.DataFrame, repaired: bool) -> float:
    rows = groups[groups["split"] == "test"]
    column = "repaired_covered_count" if repaired else "baseline_covered_count"
    return float(rows[column].sum() / rows["n"].sum())


def cell_ordering_details(cells: pd.DataFrame) -> dict[str, Any]:
    repair = cells[(cells["split"] == "repair") & cells["supported"].map(as_bool)]
    test = cells[(cells["split"] == "test") & cells["supported"].map(as_bool)]
    repair_flips: list[str] = []
    persisted: list[str] = []
    test_baseline_flips: list[str] = []
    details: list[dict[str, Any]] = []
    for cell_idx, rows in repair.groupby("cell_idx"):
        rows = rows.sort_values("group")
        base = rows["baseline_mean_set_size"].to_numpy(float)
        repaired = rows["repaired_mean_set_size"].to_numpy(float)
        base_delta = float(base[1] - base[0])
        repaired_delta = float(repaired[1] - repaired[0])
        base_sign = 0.0 if abs(base_delta) <= TOLERANCE else float(np.sign(base_delta))
        repaired_sign = 0.0 if abs(repaired_delta) <= TOLERANCE else float(np.sign(repaired_delta))
        cell_value = str(rows.iloc[0]["cell_value"])
        flipped = bool(base_sign and repaired_sign and base_sign != repaired_sign)
        if flipped:
            repair_flips.append(cell_value)
        test_rows = test[test["cell_idx"] == cell_idx].sort_values("group")
        if len(test_rows) == 2:
            test_base = test_rows["baseline_mean_set_size"].to_numpy(float)
            test_rep = test_rows["repaired_mean_set_size"].to_numpy(float)
            test_base_delta = float(test_base[1] - test_base[0])
            test_rep_delta = float(test_rep[1] - test_rep[0])
            test_base_sign = 0.0 if abs(test_base_delta) <= TOLERANCE else float(np.sign(test_base_delta))
            test_rep_sign = 0.0 if abs(test_rep_delta) <= TOLERANCE else float(np.sign(test_rep_delta))
            if test_base_sign and test_rep_sign and test_base_sign != test_rep_sign:
                test_baseline_flips.append(cell_value)
            if flipped and base_sign and test_rep_sign and base_sign != test_rep_sign:
                persisted.append(cell_value)
        else:
            test_base_delta = test_rep_delta = float("nan")
        details.append(
            {
                "cell": cell_value,
                "repair_baseline_signed_gap": base_delta,
                "repair_repaired_signed_gap": repaired_delta,
                "repair_ordering_flip": flipped,
                "test_baseline_signed_gap": test_base_delta,
                "test_repaired_signed_gap": test_rep_delta,
                "repair_flip_persists_on_test": cell_value in persisted,
            }
        )
    return {
        "repair_ordering_flip_count": len(repair_flips),
        "repair_ordering_flip_strata": json.dumps(repair_flips, separators=(",", ":")),
        "test_ordering_flip_count": len(test_baseline_flips),
        "test_ordering_flip_strata": json.dumps(test_baseline_flips, separators=(",", ":")),
        "repair_flip_persists_on_test_count": len(persisted),
        "repair_flip_persists_on_test_strata": json.dumps(persisted, separators=(",", ":")),
        "ordering_flip_details": json.dumps(details, separators=(",", ":")),
    }


def load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    names = {
        "summary": "directional_set_size_summary.csv",
        "offsets": "directional_set_size_offsets.csv",
        "cells": "directional_set_size_cell_ledger.csv",
        "groups": "directional_set_size_group_ledger.csv",
        "search": "directional_set_size_search_diagnostics.csv",
        "validation": "directional_set_size_validation.csv",
        "test": "directional_set_size_test_results.csv",
        "events": "directional_set_size_protocol_events.csv",
    }
    return {key: pd.read_csv(directory / filename) for key, filename in names.items()}


def output_set_complete(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in CORE_OUTPUTS)


def validation_records(
    dataset: str,
    score: str,
    seed: int,
    directory: Path,
    tables: dict[str, pd.DataFrame],
    done: dict[str, Any],
    run_config: dict[str, Any],
    wrapper_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: Any = "") -> None:
        records.append(
            {
                "dataset": dataset,
                "score": score,
                "resplit_id": seed - 2000,
                "resplit_seed": seed,
                "check": check,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    for row in tables["validation"].itertuples(index=False):
        add(f"runtime::{row.check}", as_bool(row.passed), row.value)
    for check, passed in wrapper_validation["checks"].items():
        add(f"wrapper::{check}", bool(passed))
    add("done_status_success", done.get("status") == "success")
    add("done_validation_pass", bool(done.get("validation_pass")))
    add("required_output_files_present", output_set_complete(directory))
    add("optimizer_seed_2026", int(run_config["optimizer_seed"]) == 2026)
    add("no_train_rows", int(run_config["train_rows_in_pool"]) == 0)
    add("base_classifier_not_retrained", not bool(run_config["base_classifier_retrained"]))
    add("test_not_used_for_selection", not bool(run_config["test_used_for_selection"]))
    add(
        "protocol_events_exact",
        tuple(tables["events"].sort_values("sequence")["event"].astype(str)) == EXPECTED_EVENTS,
    )
    return records


def per_problem_row(
    dataset: str,
    score: str,
    seed: int,
    tables: dict[str, pd.DataFrame],
    done: dict[str, Any],
    validation_pass: bool,
) -> dict[str, Any]:
    summary = tables["summary"].iloc[0]
    cells = tables["cells"]
    groups = tables["groups"]
    repair_improved, repair_tied, repair_worsened = classify_cells(cells, "repair")
    test_improved, test_tied, test_worsened = classify_cells(cells, "test")
    repair_worst_reduction = float(summary["repair_baseline_worst_gap"] - summary["repair_repaired_worst_gap"])
    repair_mean_reduction = float(summary["repair_baseline_mean_gap"] - summary["repair_repaired_mean_gap"])
    test_worst_reduction = float(summary["test_baseline_worst_gap"] - summary["test_repaired_worst_gap"])
    test_mean_reduction = float(summary["test_baseline_mean_gap"] - summary["test_repaired_mean_gap"])
    baseline_coverage = weighted_coverage(groups, False)
    repaired_coverage = weighted_coverage(groups, True)
    repair_rows = cells[(cells["split"] == "repair") & cells["supported"].map(as_bool)]
    smaller = repair_rows[repair_rows["smaller_side_budget_margin"].notna()]
    larger = repair_rows[repair_rows["larger_side_nonincrease_margin"].notna()]
    ordering = cell_ordering_details(cells)
    runtime_checks = {
        str(row.check): as_bool(row.passed)
        for row in tables["validation"].itertuples(index=False)
    }
    return {
        "dataset": dataset,
        "score": score,
        "resplit_id": seed - 2000,
        "resplit_seed": seed,
        "optimizer_seed": 2026,
        "n_repair": int(groups.loc[groups["split"] == "repair", "n"].sum()),
        "n_calibration": int(groups.loc[groups["split"] == "calibration", "n"].sum()),
        "n_test": int(groups.loc[groups["split"] == "test", "n"].sum()),
        "supported_cells": int(summary["supported_cell_count"]),
        "neutral_cells": int(summary["neutral_cell_count"]),
        "selected_theta_plus": summary["selected_theta_plus"],
        "selected_theta_minus": summary["selected_theta_minus"],
        "selected_candidate_key": summary["selected_candidate_key"],
        "selected_search_phase": summary["selected_search_phase"],
        "selected_shift_norm": float(summary["selected_shift_norm"]),
        "theta_zero_selected": float(summary["selected_shift_norm"]) == 0.0,
        "total_candidates": int(summary["candidate_count_total"]),
        "feasible_candidates": int(summary["candidate_count_feasible"]),
        "repair_worst_gap_baseline": float(summary["repair_baseline_worst_gap"]),
        "repair_worst_gap_repaired": float(summary["repair_repaired_worst_gap"]),
        "repair_worst_gap_abs_reduction": repair_worst_reduction,
        "repair_worst_gap_rel_reduction_pct": relative_percent(repair_worst_reduction, float(summary["repair_baseline_worst_gap"])),
        "repair_worst_gap_category": classify_reduction(repair_worst_reduction),
        "repair_mean_gap_baseline": float(summary["repair_baseline_mean_gap"]),
        "repair_mean_gap_repaired": float(summary["repair_repaired_mean_gap"]),
        "repair_mean_gap_abs_reduction": repair_mean_reduction,
        "repair_mean_gap_rel_reduction_pct": relative_percent(repair_mean_reduction, float(summary["repair_baseline_mean_gap"])),
        "repair_mean_gap_category": classify_reduction(repair_mean_reduction),
        "repair_total_gap_reduction": float(summary["repair_total_gap_reduction"]),
        "repair_cells_improved": repair_improved,
        "repair_cells_tied": repair_tied,
        "repair_cells_worsened": repair_worsened,
        "repair_overall_mean_set_size_baseline": float(summary["repair_baseline_overall_mean_set_size"]),
        "repair_overall_mean_set_size_repaired": float(summary["repair_repaired_overall_mean_set_size"]),
        "delta_repair_overall_mean_set_size": float(summary["repair_repaired_overall_mean_set_size"] - summary["repair_baseline_overall_mean_set_size"]),
        "repair_max_smaller_side_increase": float(smaller["mean_set_size_change"].max()) if len(smaller) else float("nan"),
        "repair_max_larger_side_change": float(larger["mean_set_size_change"].max()) if len(larger) else float("nan"),
        "test_worst_gap_baseline": float(summary["test_baseline_worst_gap"]),
        "test_worst_gap_repaired": float(summary["test_repaired_worst_gap"]),
        "test_worst_gap_abs_reduction": test_worst_reduction,
        "test_worst_gap_rel_reduction_pct": relative_percent(test_worst_reduction, float(summary["test_baseline_worst_gap"])),
        "test_worst_gap_category": classify_reduction(test_worst_reduction),
        "test_mean_gap_baseline": float(summary["test_baseline_mean_gap"]),
        "test_mean_gap_repaired": float(summary["test_repaired_mean_gap"]),
        "test_mean_gap_abs_reduction": test_mean_reduction,
        "test_mean_gap_rel_reduction_pct": relative_percent(test_mean_reduction, float(summary["test_baseline_mean_gap"])),
        "test_mean_gap_category": classify_reduction(test_mean_reduction),
        "test_total_gap_reduction": float(summary["test_total_gap_reduction"]),
        "test_cells_improved": test_improved,
        "test_cells_tied": test_tied,
        "test_cells_worsened": test_worsened,
        "overall_mean_set_size_baseline": float(summary["test_baseline_overall_mean_set_size"]),
        "overall_mean_set_size_repaired": float(summary["test_repaired_overall_mean_set_size"]),
        "delta_overall_mean_set_size": float(summary["test_repaired_overall_mean_set_size"] - summary["test_baseline_overall_mean_set_size"]),
        "test_overall_coverage_baseline": baseline_coverage,
        "test_overall_coverage_repaired": repaired_coverage,
        "delta_overall_coverage": repaired_coverage - baseline_coverage,
        "test_worst_coverage_gap_baseline": float(summary["test_baseline_worst_coverage_gap"]),
        "test_worst_coverage_gap_repaired": float(summary["test_repaired_worst_coverage_gap"]),
        "delta_worst_coverage_gap": float(summary["test_repaired_worst_coverage_gap"] - summary["test_baseline_worst_coverage_gap"]),
        "test_mean_coverage_gap_baseline": float(summary["test_baseline_mean_coverage_gap"]),
        "test_mean_coverage_gap_repaired": float(summary["test_repaired_mean_coverage_gap"]),
        "delta_mean_coverage_gap": float(summary["test_repaired_mean_coverage_gap"] - summary["test_baseline_mean_coverage_gap"]),
        "test_max_abs_cell_coverage_change": float(summary["test_maximum_absolute_cellwise_coverage_change"]),
        **ordering,
        "repair_exact_group_counts_pass": as_bool(summary["repair_group_covered_counts_exact"]),
        "calibration_exact_group_counts_pass": as_bool(summary["calibration_group_covered_counts_exact"]),
        "all_cell_feasibility_pass": bool(
            runtime_checks.get("all_repair_feasibility_constraints", False)
        ),
        "protocol_isolation_pass": tuple(tables["events"].sort_values("sequence")["event"].astype(str)) == EXPECTED_EVENTS,
        "required_output_files_present": output_set_complete(
            problem_directory(seed, dataset, score)
        ),
        "validation_pass": validation_pass,
        "runtime_seconds": float(done["runtime_seconds"]),
    }


def frames_equal(left: pd.DataFrame, right: pd.DataFrame, drop: Iterable[str] = ()) -> bool:
    columns = [column for column in left.columns if column not in set(drop)]
    if columns != [column for column in right.columns if column not in set(drop)]:
        return False
    if left.shape[0] != right.shape[0]:
        return False
    for column in columns:
        a = left[column]
        b = right[column]
        numeric_a = pd.to_numeric(a, errors="coerce")
        numeric_b = pd.to_numeric(b, errors="coerce")
        numeric = bool(
            np.array_equal(a.isna().to_numpy(), b.isna().to_numpy())
            and numeric_a[a.notna()].notna().all()
            and numeric_b[b.notna()].notna().all()
        )
        if numeric:
            if not np.array_equal(numeric_a.to_numpy(float), numeric_b.to_numpy(float), equal_nan=True):
                return False
        elif not np.array_equal(
            a.fillna("<NA>").astype(str).to_numpy(),
            b.fillna("<NA>").astype(str).to_numpy(),
        ):
            return False
    return True


def folktables_b_identity(seed: int) -> tuple[bool, str]:
    dataset = "folktables_design_b"
    aps = load_tables(problem_directory(seed, dataset, "aps"))
    raps = load_tables(problem_directory(seed, dataset, "raps"))
    summary_diagnostic_columns = (
        "score_method",
        "rejected_by_group_coverage_count",
        "rejected_by_larger_side_count",
    )
    result_tables_equal = frames_equal(
        aps["summary"], raps["summary"], drop=summary_diagnostic_columns
    ) and all(
        frames_equal(aps[key], raps[key], drop=("score_method",))
        for key in ("offsets", "cells", "groups", "validation", "test", "events")
    )
    pool, manifest = load_pool(dataset)
    indices, _, _ = load_partition(dataset, seed, pool, manifest)
    data = build_loaded_dataset(dataset, seed, pool, manifest, indices)
    score_equal = True
    for split in ("repair", "calibration", "test"):
        probs = data.splits[split].probs
        aps_scores = labelwise_scores_from_probs(probs, "aps", raps_lambda=0.01, raps_k_reg=5, saps_lambda=0.2)
        raps_scores = labelwise_scores_from_probs(probs, "raps", raps_lambda=0.01, raps_k_reg=5, saps_lambda=0.2)
        score_equal = score_equal and bool(np.array_equal(aps_scores, raps_scores))
    return bool(score_equal and result_tables_equal), (
        f"baseline_scores={score_equal}; selected_scientific_outputs="
        f"{result_tables_equal}; unselected_candidate_rejection_diagnostics="
        "excluded"
    )


def quantile_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return {
        "valid_n": int(len(numeric)),
        "mean": float(numeric.mean()) if len(numeric) else float("nan"),
        "sd": float(numeric.std(ddof=1)) if len(numeric) > 1 else float("nan"),
        "median": float(numeric.median()) if len(numeric) else float("nan"),
        "q1": float(numeric.quantile(0.25)) if len(numeric) else float("nan"),
        "q3": float(numeric.quantile(0.75)) if len(numeric) else float("nan"),
        "minimum": float(numeric.min()) if len(numeric) else float("nan"),
        "maximum": float(numeric.max()) if len(numeric) else float("nan"),
    }


def primary_summary(per_problem: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, score), group in per_problem.groupby(["dataset", "score"], sort=False):
        repair_worst = group["repair_worst_gap_category"].value_counts()
        test_worst = group["test_worst_gap_category"].value_counts()
        test_mean = group["test_mean_gap_category"].value_counts()
        repair_dist = quantile_summary(group["repair_worst_gap_abs_reduction"])
        test_dist = quantile_summary(group["test_worst_gap_abs_reduction"])
        rows.append(
            {
                "dataset": dataset,
                "score": score,
                "valid_resplits": int(group["validation_pass"].sum()),
                "repair_worst_improved_n": int(repair_worst.get("improved", 0)),
                "repair_worst_tied_n": int(repair_worst.get("tied", 0)),
                "repair_worst_worsened_n": int(repair_worst.get("worsened", 0)),
                "repair_mean_improved_n": int((group["repair_mean_gap_category"] == "improved").sum()),
                "repair_worst_abs_reduction_median": repair_dist["median"],
                "repair_worst_abs_reduction_q1": repair_dist["q1"],
                "repair_worst_abs_reduction_q3": repair_dist["q3"],
                "test_worst_improved_n": int(test_worst.get("improved", 0)),
                "test_worst_tied_n": int(test_worst.get("tied", 0)),
                "test_worst_worsened_n": int(test_worst.get("worsened", 0)),
                "test_worst_abs_reduction_median": test_dist["median"],
                "test_worst_abs_reduction_q1": test_dist["q1"],
                "test_worst_abs_reduction_q3": test_dist["q3"],
                "test_mean_improved_n": int(test_mean.get("improved", 0)),
                "test_mean_tied_n": int(test_mean.get("tied", 0)),
                "test_mean_worsened_n": int(test_mean.get("worsened", 0)),
                "theta_zero_n": int(group["theta_zero_selected"].sum()),
                "ordering_flip_resplit_n": int((group["repair_ordering_flip_count"] > 0).sum()),
                "delta_test_mean_set_size_median": float(group["delta_overall_mean_set_size"].median()),
                "delta_overall_coverage_median": float(group["delta_overall_coverage"].median()),
                "delta_worst_coverage_gap_median": float(group["delta_worst_coverage_gap"].median()),
                "worst_coverage_gap_increase_n": int((group["delta_worst_coverage_gap"] > TOLERANCE).sum()),
                "worst_coverage_gap_increase_gt_0_01_n": int((group["delta_worst_coverage_gap"] > 0.01).sum()),
                "worst_coverage_gap_increase_gt_0_05_n": int((group["delta_worst_coverage_gap"] > 0.05).sum()),
                "max_cell_coverage_move_gt_0_01_n": int((group["test_max_abs_cell_coverage_change"] > 0.01).sum()),
                "max_cell_coverage_move_gt_0_05_n": int((group["test_max_abs_cell_coverage_change"] > 0.05).sum()),
                "validation_pass": bool(group["validation_pass"].all() and len(group) == 20),
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(per_problem: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", per_problem)]
    scopes.extend(("dataset", str(key), frame) for key, frame in per_problem.groupby("dataset"))
    scopes.extend(("score", str(key), frame) for key, frame in per_problem.groupby("score"))
    scopes.extend(
        ("dataset_score", f"{dataset}/{score}", frame)
        for (dataset, score), frame in per_problem.groupby(["dataset", "score"])
    )
    for scope, key, frame in scopes:
        for metric in (*ABSOLUTE_METRICS, *RELATIVE_METRICS):
            rows.append(
                {
                    "scope": scope,
                    "scope_value": key,
                    "metric": metric,
                    "total_n": len(frame),
                    **quantile_summary(frame[metric]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_frequency_summary(per_problem: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", per_problem)]
    scopes.extend(("dataset", str(key), frame) for key, frame in per_problem.groupby("dataset"))
    scopes.extend(("score", str(key), frame) for key, frame in per_problem.groupby("score"))
    for scope, key, frame in scopes:
        rows.append(
            {
                "scope": scope,
                "scope_value": key,
                "problems": len(frame),
                "valid": int(frame["validation_pass"].sum()),
                "invalid": int((~frame["validation_pass"]).sum()),
                "repair_exact_count_pass": int(frame["repair_exact_group_counts_pass"].sum()),
                "calibration_exact_count_pass": int(frame["calibration_exact_group_counts_pass"].sum()),
                "all_cell_feasibility_pass": int(frame["all_cell_feasibility_pass"].sum()),
                "protocol_isolation_pass": int(frame["protocol_isolation_pass"].sum()),
                "theta_zero": int(frame["theta_zero_selected"].sum()),
                "theta_nonzero": int((~frame["theta_zero_selected"]).sum()),
                "repair_worst_improved": int((frame["repair_worst_gap_category"] == "improved").sum()),
                "repair_mean_improved": int((frame["repair_mean_gap_category"] == "improved").sum()),
                "test_worst_improved": int((frame["test_worst_gap_category"] == "improved").sum()),
                "test_worst_tied": int((frame["test_worst_gap_category"] == "tied").sum()),
                "test_worst_worsened": int((frame["test_worst_gap_category"] == "worsened").sum()),
                "test_mean_improved": int((frame["test_mean_gap_category"] == "improved").sum()),
                "test_mean_tied": int((frame["test_mean_gap_category"] == "tied").sum()),
                "test_mean_worsened": int((frame["test_mean_gap_category"] == "worsened").sum()),
                "ordering_flip_problems": int((frame["repair_ordering_flip_count"] > 0).sum()),
                "worst_coverage_gap_increase": int((frame["delta_worst_coverage_gap"] > TOLERANCE).sum()),
                "worst_coverage_gap_increase_gt_0_01": int((frame["delta_worst_coverage_gap"] > 0.01).sum()),
                "worst_coverage_gap_increase_gt_0_05": int((frame["delta_worst_coverage_gap"] > 0.05).sum()),
                "max_cell_coverage_move_gt_0_01": int((frame["test_max_abs_cell_coverage_change"] > 0.01).sum()),
                "max_cell_coverage_move_gt_0_05": int((frame["test_max_abs_cell_coverage_change"] > 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def canonical_comparison(
    per_problem: pd.DataFrame, canonical_summary: Path | None
) -> pd.DataFrame:
    if canonical_summary is None:
        return pd.DataFrame()
    canonical = pd.read_csv(canonical_summary)
    metrics = (
        "repair_worst_gap_abs_reduction",
        "repair_mean_gap_abs_reduction",
        "test_worst_gap_abs_reduction",
        "test_mean_gap_abs_reduction",
        "delta_overall_mean_set_size",
        "delta_test_overall_coverage",
        "delta_test_worst_coverage_gap",
        "test_max_abs_cell_coverage_change",
    )
    robust_names = {
        "delta_test_overall_coverage": "delta_overall_coverage",
        "delta_test_worst_coverage_gap": "delta_worst_coverage_gap",
    }
    rows: list[dict[str, Any]] = []
    for canonical_row in canonical.itertuples(index=False):
        group = per_problem[
            (per_problem["dataset"] == canonical_row.dataset)
            & (per_problem["score"] == canonical_row.score)
        ]
        for canonical_metric in metrics:
            robust_metric = robust_names.get(canonical_metric, canonical_metric)
            values = group[robust_metric].astype(float)
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            value = float(getattr(canonical_row, canonical_metric))
            if value < q1:
                location = "below Q1"
            elif value <= median:
                location = "Q1 to median"
            elif value <= q3:
                location = "median to Q3"
            else:
                location = "above Q3"
            rows.append(
                {
                    "dataset": canonical_row.dataset,
                    "score": canonical_row.score,
                    "metric": robust_metric,
                    "canonical_value": value,
                    "resplit_q1": q1,
                    "resplit_median": median,
                    "resplit_q3": q3,
                    "canonical_location": location,
                    "canonical_within_resplit_iqr": bool(q1 <= value <= q3),
                }
            )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Score | N | Repair worst I/T/W | Repair median [IQR] | Test worst I/T/W | Test median [IQR] | Test mean I/T/W | Theta zero | Flip resplits | Coverage-gap increase | Validation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[row.dataset],
                    row.score.upper(),
                    str(row.valid_resplits),
                    f"{row.repair_worst_improved_n}/{row.repair_worst_tied_n}/{row.repair_worst_worsened_n}",
                    f"{fmt(row.repair_worst_abs_reduction_median)} [{fmt(row.repair_worst_abs_reduction_q1)}, {fmt(row.repair_worst_abs_reduction_q3)}]",
                    f"{row.test_worst_improved_n}/{row.test_worst_tied_n}/{row.test_worst_worsened_n}",
                    f"{fmt(row.test_worst_abs_reduction_median)} [{fmt(row.test_worst_abs_reduction_q1)}, {fmt(row.test_worst_abs_reduction_q3)}]",
                    f"{row.test_mean_improved_n}/{row.test_mean_tied_n}/{row.test_mean_worsened_n}",
                    str(row.theta_zero_n),
                    str(row.ordering_flip_resplit_n),
                    str(row.worst_coverage_gap_increase_n),
                    "PASS" if row.validation_pass else "FAIL",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    per_problem: pd.DataFrame,
    summary: pd.DataFrame,
    distributions: pd.DataFrame,
    aggregate: pd.DataFrame,
    canonical: pd.DataFrame,
    identity: pd.DataFrame,
) -> None:
    overall = aggregate[(aggregate["scope"] == "overall")].iloc[0]
    negative = per_problem[per_problem["test_worst_gap_category"] != "improved"]
    lines = [
        "# Raw Set-Size Repair: 20-Resplit Robustness",
        "",
        "## Status",
        "",
        f"**{'PASS' if int(overall.invalid) == 0 and identity['passed'].all() else 'FAIL'}.** "
        "All prespecified fixed-model downstream resplits are retained. This is descriptive empirical variability, not 320 independent samples for inferential testing.",
        "",
        "## Prespecified Resplit Protocol",
        "",
        "The cohort uses 20 resplits per dataset-score configuration. Calibration, repair, and test rows form each dataset's nontraining pool; train rows are excluded. BioBias uses joint G-by-Y stratification, Folktables Design A uses its configured Y/G/Z fallback hierarchy, Folktables Design B uses joint G-by-Z stratification, and ATUS uses one deterministic respondent shuffle. Exact split sizes are preserved.",
        "",
        "No model was retrained. Each dataset-seed partition is shared across all four scores. Support and larger/smaller roles are redefined from that resplit's repair split only.",
        "",
        "## Set-Size Protocol",
        "",
        "* `alpha = 0.10`; minimum group-by-cell repair count `30`",
        "* set-size tolerance `1e-9`; smaller-side per-cell budget `0.02` labels",
        "* deterministic SIMPLE, APS, RAPS (`lambda=0.01`, `k_reg=5`), SAPS (`lambda=0.2`)",
        "* optimizer seed `2026` and fixed search budget",
        "* objective: worst gap, mean gap, total gap reduction, shift norm",
        "* exact repair and calibration coarse-group covered counts",
        "* repair-only selection, frozen theta, independent calibration, one-time test evaluation",
        "",
        "## Validation",
        "",
        f"Valid problems: **{int(overall.valid)}/{int(overall.problems)}**. Exact repair counts: **{int(overall.repair_exact_count_pass)}/{int(overall.problems)}**. Exact calibration counts: **{int(overall.calibration_exact_count_pass)}/{int(overall.problems)}**. All-cell feasibility: **{int(overall.all_cell_feasibility_pass)}/{int(overall.problems)}**. Protocol isolation: **{int(overall.protocol_isolation_pass)}/{int(overall.problems)}**.",
        "",
        f"Folktables Design B APS/RAPS identity: **{'PASS' if identity['passed'].all() else 'FAIL'}** for all 20 resplits.",
        "",
        "## Dataset-by-Score Summary",
        "",
        markdown_summary(summary),
        "",
        "I/T/W denotes improved/tied/worsened using the frozen `1e-9` tolerance. Absolute set-size effects are mean-set-size units (labels). Coverage deltas are proportions.",
        "",
        "## Aggregate Outcomes",
        "",
        f"Theta zero was selected in **{int(overall.theta_zero)}/{int(overall.problems)}** problems. Worst repair gap improved in **{int(overall.repair_worst_improved)}/{int(overall.problems)}** and mean repair gap improved in **{int(overall.repair_mean_improved)}/{int(overall.problems)}**.",
        "",
        f"Held-out worst gap improved/tied/worsened in **{int(overall.test_worst_improved)}/{int(overall.test_worst_tied)}/{int(overall.test_worst_worsened)}** problems. Held-out mean gap improved/tied/worsened in **{int(overall.test_mean_improved)}/{int(overall.test_mean_tied)}/{int(overall.test_mean_worsened)}** problems.",
        "",
        f"Repair ordering flips occurred in **{int(overall.ordering_flip_problems)}/{int(overall.problems)}** problems. Worst held-out cellwise coverage gap increased in **{int(overall.worst_coverage_gap_increase)}/{int(overall.problems)}**, exceeded a `0.01` increase in **{int(overall.worst_coverage_gap_increase_gt_0_01)}**, and exceeded `0.05` in **{int(overall.worst_coverage_gap_increase_gt_0_05)}**. Maximum absolute cellwise coverage movement exceeded `0.01` in **{int(overall.max_cell_coverage_move_gt_0_01)}** and `0.05` in **{int(overall.max_cell_coverage_move_gt_0_05)}** problems.",
        "",
        "## Effect Distributions",
        "",
        "Mean, sample SD, median, quartiles, minima, and maxima for all prespecified absolute and relative metrics are stored in `set_size_20_resplit_distribution_summary.csv`. Relative summaries exclude only undefined denominators and report their valid N.",
        "",
        "## Negative and Null Cases",
        "",
        f"All null and unfavorable held-out outcomes are retained. There are **{len(negative)}** problems whose worst held-out gap did not improve beyond tolerance. No resplit, score, or dataset was filtered based on repair or test outcomes.",
        "",
        "## Scope",
        "",
        "These 20 resplits measure empirical variability conditional on frozen trained predictors and frozen nontraining pools. They are not independent model-training runs or independent-data confidence intervals, and the 320 dataset-score-resplit problems are not treated as independent samples for formal inference.",
    ]
    if not canonical.empty:
        canonical_test = canonical[
            canonical["metric"] == "test_worst_gap_abs_reduction"
        ]
        lines.extend(
            [
                "",
                "## Canonical Context",
                "",
                f"For held-out worst-gap reduction, **{int(canonical_test['canonical_within_resplit_iqr'].sum())}/16** canonical values lie within the corresponding resplit IQR. The canonical split is not included in, and did not tune, the 20-resplit distribution.",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_identity_validation(
    validation: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        {
            "dataset": "folktables_design_b",
            "score": "aps_raps_identity",
            "resplit_id": int(row.resplit_seed) - 2000,
            "resplit_seed": int(row.resplit_seed),
            "check": "folktables_b_aps_raps_identity",
            "passed": bool(row.passed),
            "detail": row.detail,
        }
        for row in identity.itertuples(index=False)
    ]
    return pd.concat([validation, pd.DataFrame(rows)], ignore_index=True)


def aggregate(canonical_summary: Path | None = None) -> None:
    problem_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    offset_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    split_records: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in RESPLIT_SEEDS:
        for dataset in DATASETS:
            for score in SCORES:
                directory = problem_directory(seed, dataset, score)
                done_path = directory / "DONE.json"
                if not done_path.exists():
                    raise FileNotFoundError(f"Missing completed problem: {directory}")
                done = json.loads(done_path.read_text(encoding="utf-8"))
                run_config = json.loads((directory / "resplit_run_config.json").read_text(encoding="utf-8"))
                wrapper_validation = json.loads((directory / "resplit_validation.json").read_text(encoding="utf-8"))
                tables = load_tables(directory)
                checks = validation_records(
                    dataset, score, seed, directory, tables, done, run_config, wrapper_validation
                )
                validation_rows.extend(checks)
                passed = all(row["passed"] for row in checks)
                problem_rows.append(per_problem_row(dataset, score, seed, tables, done, passed))
                for key, target in (("cells", cell_frames), ("groups", group_frames), ("offsets", offset_frames), ("events", event_frames)):
                    frame = tables[key].copy()
                    frame.insert(0, "resplit_seed", seed)
                    frame.insert(0, "resplit_id", seed - 2000)
                    target.append(frame)
                partition = run_config["partition_audit"]
                split_records[(dataset, seed)] = {
                    **partition,
                    "all_four_scores_share_partition": True,
                    "score_count": 4,
                }
    per_problem = pd.DataFrame(problem_rows)
    order_dataset = pd.Categorical(per_problem["dataset"], categories=DATASETS, ordered=True)
    order_score = pd.Categorical(per_problem["score"], categories=SCORES, ordered=True)
    per_problem = per_problem.assign(_dataset=order_dataset, _score=order_score).sort_values(
        ["_dataset", "_score", "resplit_seed"]
    ).drop(columns=["_dataset", "_score"]).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows)

    identity_rows = []
    for seed in RESPLIT_SEEDS:
        passed, detail = folktables_b_identity(seed)
        identity_rows.append({"resplit_seed": seed, "passed": passed, "detail": detail})
    identity = pd.DataFrame(identity_rows)
    validation = add_identity_validation(validation, identity)
    if len(per_problem) != 320 or not per_problem["validation_pass"].all():
        raise AssertionError("Not all 320 repair problems passed validation")
    if not validation["passed"].map(as_bool).all():
        failed = validation[~validation["passed"].map(as_bool)]
        raise AssertionError(f"Aggregate validation failed:\n{failed.to_string(index=False)}")
    if not identity["passed"].all():
        raise AssertionError("Folktables Design B APS/RAPS identity failed")

    summary = primary_summary(per_problem)
    distributions = distribution_summary(per_problem)
    aggregate_summary = aggregate_frequency_summary(per_problem)
    canonical = canonical_comparison(per_problem, canonical_summary)
    cells = pd.concat(cell_frames, ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    offsets = pd.concat(offset_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    split_manifest = pd.DataFrame(list(split_records.values())).sort_values(
        ["dataset", "resplit_seed"]
    )
    coverage = per_problem[
        [
            "dataset",
            "score",
            "resplit_id",
            "resplit_seed",
            "test_overall_coverage_baseline",
            "test_overall_coverage_repaired",
            "delta_overall_coverage",
            "test_worst_coverage_gap_baseline",
            "test_worst_coverage_gap_repaired",
            "delta_worst_coverage_gap",
            "test_mean_coverage_gap_baseline",
            "test_mean_coverage_gap_repaired",
            "delta_mean_coverage_gap",
            "test_max_abs_cell_coverage_change",
        ]
    ].copy()

    per_problem.to_csv(RESULT_ROOT / "set_size_20_resplit_per_problem.csv", index=False)
    summary.to_csv(RESULT_ROOT / "set_size_20_resplit_summary_16.csv", index=False)
    cells.to_csv(RESULT_ROOT / "set_size_20_resplit_cell_ledger.csv", index=False)
    groups.to_csv(RESULT_ROOT / "set_size_20_resplit_group_ledger.csv", index=False)
    offsets.to_csv(RESULT_ROOT / "set_size_20_resplit_offsets.csv", index=False)
    coverage.to_csv(RESULT_ROOT / "set_size_20_resplit_coverage_diagnostics.csv", index=False)
    validation.to_csv(RESULT_ROOT / "set_size_20_resplit_validation.csv", index=False)
    events.to_csv(RESULT_ROOT / "set_size_20_resplit_protocol_events.csv", index=False)
    split_manifest.to_csv(RESULT_ROOT / "set_size_20_resplit_split_manifest.csv", index=False)
    distributions.to_csv(RESULT_ROOT / "set_size_20_resplit_distribution_summary.csv", index=False)
    if not canonical.empty:
        canonical.to_csv(
            RESULT_ROOT / "set_size_20_resplit_canonical_comparison.csv", index=False
        )
    aggregate_summary.to_csv(RESULT_ROOT / "set_size_20_resplit_aggregate_summary.csv", index=False)
    identity.to_csv(RESULT_ROOT / "folktables_b_aps_raps_identity.csv", index=False)

    config = {
        "experiment": "raw_set_size_20_resplit_robustness",
        "datasets": list(DATASETS),
        "scores": list(SCORES),
        "resplit_ids": list(range(1, 21)),
        "resplit_seeds": list(RESPLIT_SEEDS),
        "problem_count": 320,
        "alpha": 0.10,
        "min_cell_count": 30,
        "set_size_tolerance": 1e-9,
        "smaller_side_budget": 0.02,
        "raps_lambda": 0.01,
        "raps_k_reg": 5,
        "saps_lambda": 0.2,
        "optimizer_seed": 2026,
        "selection_split": "repair",
        "calibration_split": "calibration",
        "evaluation_split": "test",
        "fixed_model": True,
        "folktables_b_aps_raps_identity_pass": bool(identity["passed"].all()),
        "descriptive_only": True,
    }
    write_json(RESULT_ROOT / "set_size_20_resplit_config.json", config)
    write_report(
        per_problem,
        summary,
        distributions,
        aggregate_summary,
        canonical,
        identity,
    )
    print(markdown_summary(summary))
    print(f"REPORT={REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the 20-resplit set-size experiment."
    )
    parser.add_argument("--stage", choices=("aggregate",), required=True)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=robustness_runner.DEFAULT_PROTOCOL_ROOT,
    )
    parser.add_argument(
        "--output-root", type=Path, default=robustness_runner.DEFAULT_RESULT_ROOT
    )
    parser.add_argument(
        "--canonical-summary",
        type=Path,
        help="Optional canonical 16-row summary for distribution context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.protocol_root, args.output_root)
    aggregate(args.canonical_summary)


if __name__ == "__main__":
    main()
