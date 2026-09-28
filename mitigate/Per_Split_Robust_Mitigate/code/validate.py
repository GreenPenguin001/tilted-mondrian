"""End-to-end scientific validation for the Table 2 resplits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiment import REPORTING_SPLITS
from .source_registry import (
    COMPONENT_ROOT,
    DATASETS,
    SCORES,
    configuration_split_seeds,
    configured_configuration_keys,
    output_root,
)


TOLERANCE = 1e-9
LEDGER_TOLERANCE = 1e-12
SUMMARY_FILES = (
    "table2_all_scores_per_split_long.csv",
    "table2_all_scores_split_summary.csv",
    "table2_all_scores_dataset_summary.csv",
    "table2_all_scores_score_summary.csv",
    "table2_all_scores_constraint_summary.csv",
    "table2_all_scores_cell_ledger_long.csv",
    "table2_all_scores_group_ledger_long.csv",
    "table2_all_scores_role_change_summary.csv",
    "table2_all_scores_runtime_summary.csv",
    "table2_all_scores_failure_summary.csv",
    "table2_all_scores_appendix_ready.csv",
    "table2_all_scores_appendix_ready.tex",
    "table2_all_scores_robustness.pdf",
)


def boolean_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    detail: str,
) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def close(left: Any, right: Any, tolerance: float = LEDGER_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def validate_configuration(
    row: pd.Series,
    cells: pd.DataFrame,
    groups: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    dataset = str(row["dataset"])
    score = str(row["score"])
    seed = int(row["split_seed"])
    key = (dataset, score, seed)
    selected_cells = cells.loc[
        (cells["dataset"].astype(str) == dataset)
        & (cells["score"].astype(str) == score)
        & (cells["split_seed"].astype(int) == seed)
    ]
    selected_groups = groups.loc[
        (groups["dataset"].astype(str) == dataset)
        & (groups["score"].astype(str) == score)
        & (groups["split_seed"].astype(int) == seed)
    ]
    test = selected_cells.loc[
        (selected_cells["split"].astype(str) == "test")
        & selected_cells["repairable"].map(boolean_value)
    ]
    repair = selected_cells.loc[
        (selected_cells["split"].astype(str) == "repair")
        & selected_cells["repairable"].map(boolean_value)
    ]
    if test.empty or repair.empty:
        return [f"{key}: missing repairable repair/test ledger rows"]

    test_by_cell = test.groupby(["cell_id", "cell_name"], sort=False).first().reset_index()
    baseline_max = float(test_by_cell["baseline_absolute_local_gap"].max())
    directional_max = float(test_by_cell["directional_absolute_local_gap"].max())
    maximum_mitigation = float(test_by_cell["local_gap_mitigation"].max())
    maximum_cell = str(
        test_by_cell.loc[
            test_by_cell["local_gap_mitigation"].astype(float).idxmax(), "cell_name"
        ]
    )
    reconstructed = {
        "baseline_test_max_gap": baseline_max,
        "directional_test_max_gap": directional_max,
        "directional_test_delta_max_gap": baseline_max - directional_max,
        "heldout_maximum_single_cell_mitigation": maximum_mitigation,
    }
    for field, expected in reconstructed.items():
        if not close(row[field], expected):
            errors.append(f"{key}: {field} summary={row[field]} ledger={expected}")
    if str(row["heldout_maximum_mitigation_cell"]) != maximum_cell:
        errors.append(
            f"{key}: heldout argmax={row['heldout_maximum_mitigation_cell']} ledger={maximum_cell}"
        )

    repair_advantaged = repair.loc[
        repair["repair_defined_role"].astype(str) == "advantaged"
    ]
    harm = float(max(0.0, (-repair_advantaged["coverage_change"].astype(float)).max()))
    if not close(row["maximum_advantaged_harm_repair"], harm):
        errors.append(
            f"{key}: repair harm summary={row['maximum_advantaged_harm_repair']} ledger={harm}"
        )

    test_groups = selected_groups.loc[selected_groups["split"].astype(str) == "test"]
    set_size_change = float(
        (
            test_groups["set_size_change"].astype(float)
            * test_groups["n"].astype(float)
        ).sum()
        / test_groups["n"].astype(float).sum()
    )
    if not close(row["test_average_set_size_change"], set_size_change):
        errors.append(
            f"{key}: test set-size summary={row['test_average_set_size_change']} ledger={set_size_change}"
        )

    for field, split in REPORTING_SPLITS.items():
        if str(row[field]) != split:
            errors.append(f"{key}: {field}={row[field]!r}, expected {split!r}")

    for split in ("repair", "calibration"):
        split_groups = selected_groups.loc[selected_groups["split"].astype(str) == split]
        exact = bool(
            (
                split_groups["baseline_covered_count"].astype(int)
                == split_groups["repaired_covered_count"].astype(int)
            ).all()
        )
        if not exact:
            errors.append(f"{key}: {split} group covered counts are not exact")

    if float(row["maximum_advantaged_harm_repair"]) > 0.01 + TOLERANCE:
        errors.append(f"{key}: advantaged harm exceeds epsilon")
    if float(row["minimum_disadvantaged_cell_change_repair"]) < -TOLERANCE:
        errors.append(f"{key}: disadvantaged repair cell worsened")
    if float(row["maximum_local_gap_increase_repair"]) > TOLERANCE:
        errors.append(f"{key}: repair local gap increased")
    required_true = (
        "repair_group_covered_counts_exact",
        "calibration_group_covered_counts_exact",
        "every_repair_constraint_pass",
        "symmetric_candidate_flag",
        "symmetric_candidate_equivalence_flag",
        "directional_repair_not_worse_than_symmetric_flag",
    )
    for field in required_true:
        if not boolean_value(row[field]):
            errors.append(f"{key}: {field} is not true")
    configuration = output_root() / "splits" / f"seed_{seed}" / dataset / score
    local_checks = pd.read_csv(
        configuration / "validation" / "reporting_source_checks.csv"
    )
    if not local_checks["passed"].map(boolean_value).all():
        errors.append(f"{key}: local reporting-source checks failed")
    run_config = json.loads(
        (configuration / "run_config.json").read_text(encoding="utf-8")
    )
    if run_config.get("test_used_for_selection") is not False:
        errors.append(f"{key}: run config does not certify test exclusion")
    if run_config.get("base_classifier_retrained") is not False:
        errors.append(f"{key}: run config reports base retraining")
    if int(run_config.get("train_rows_in_pool", -1)) != 0:
        errors.append(f"{key}: run config reports train contamination")
    if run_config.get("optimizer_seed") != 2026:
        errors.append(f"{key}: optimizer seed changed")
    return errors


def validate_partitions(
    rows: list[dict[str, Any]],
    per_split: pd.DataFrame,
    expected_by_configuration: dict[tuple[str, str], list[int]],
) -> None:
    expected_by_dataset = {
        specification.dataset: sorted(
            {
                seed
                for score in SCORES
                for seed in expected_by_configuration[(specification.dataset, score)]
            }
        )
        for specification in DATASETS
    }
    for specification in DATASETS:
        dataset = specification.dataset
        for seed in expected_by_dataset[dataset]:
            manifest_path = (
                output_root()
                / "split_manifests"
                / f"seed_{seed}"
                / dataset
                / "split_manifest.json"
            )
            indices_path = manifest_path.with_name("split_indices.npz")
            if not manifest_path.exists() or not indices_path.exists():
                add_check(
                    rows, f"partition_{dataset}_{seed}", False, "manifest or indices missing"
                )
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrays = np.load(indices_path)
            pool_manifest = json.loads(
                (output_root() / "pools" / dataset / "pool_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            all_indices = np.concatenate(
                [arrays[split] for split in ("calibration", "repair", "test")]
            )
            passed = (
                int(manifest["train_rows"]) == 0
                and bool(manifest["disjoint"])
                and bool(manifest["union_is_full_pool"])
                and len(all_indices) == int(pool_manifest["n_pool"])
                and len(np.unique(all_indices)) == len(all_indices)
                and set(all_indices.tolist()) == set(range(int(pool_manifest["n_pool"])))
                and all(
                    len(arrays[split])
                    == int(pool_manifest["canonical_split_sizes"][split])
                    for split in ("calibration", "repair", "test")
                )
            )
            add_check(
                rows,
                f"partition_{dataset}_{seed}",
                passed,
                f"n={len(all_indices)}, train_rows={manifest['train_rows']}",
            )

    shared = True
    failures = []
    failed_keys = set()
    configured = configured_configuration_keys()
    for path in (output_root() / "splits").glob("seed_*/*/*__FAILED_*/FAILED.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(payload["dataset"]),
            str(payload["score"]),
            int(payload["split_seed"]),
        )
        if key in configured:
            failed_keys.add((key[0], key[2], key[1]))
    for (dataset, seed), group in per_split.groupby(
        ["dataset", "split_seed"], sort=False
    ):
        expected_manifest = (
            output_root()
            / "split_manifests"
            / f"seed_{int(seed)}"
            / str(dataset)
            / "split_manifest.json"
        )
        for score in group["score"].astype(str):
            run_config_path = (
                output_root()
                / "splits"
                / f"seed_{int(seed)}"
                / str(dataset)
                / score
                / "run_config.json"
            )
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            configured_manifest = Path(str(run_config["partition_manifest"]))
            if configured_manifest.resolve() != expected_manifest.resolve():
                shared = False
                failures.append(f"{dataset}:{seed}:{score}:partition_manifest")
        expected_scores = {
            score
            for score in SCORES
            if int(seed) in expected_by_configuration[(str(dataset), score)]
        }
        missing_scores = expected_scores - set(group["score"].astype(str))
        unaccounted = [
            score
            for score in missing_scores
            if (str(dataset), int(seed), score) not in failed_keys
        ]
        if unaccounted:
            shared = False
            failures.append(
                f"{dataset}:{seed}:unaccounted_scores={','.join(sorted(unaccounted))}"
            )
    add_check(
        rows,
        "all_scores_share_dataset_seed_partition",
        shared,
        "all configurations reference the same validated split indices"
        if shared
        else ";".join(failures),
    )


def write_validation_report(
    checks: pd.DataFrame,
    configuration_errors: list[str],
) -> None:
    passed = bool(checks["passed"].map(boolean_value).all() and not configuration_errors)
    lines = [
        "# Table 2 All-Score Validation Report",
        "",
        f"Overall status: **{'PASS' if passed else 'FAIL'}**",
        "",
        "This is fixed-model random-resplit empirical variability, not independent model-training runs and not independent-data confidence intervals.",
        "",
        "## Global checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for row in checks.to_dict("records"):
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| {row['check']} | {row['passed']} | {detail} |")
    lines.extend(["", "## Configuration errors", ""])
    if configuration_errors:
        lines.extend(f"- {error}" for error in configuration_errors)
    else:
        lines.append(
            "None. Every completed configuration was independently reconstructed from raw ledgers."
        )
    lines.extend(
        [
            "",
            "All held-out Table 2 fields were reconstructed from test ledgers. Harm and feasibility were reconstructed from repair ledgers. Calibration was checked only for exact covered-count matching after theta was frozen. Negative held-out outcomes were retained and were not treated as failures.",
        ]
    )
    (output_root() / "VALIDATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    payload = {
        "passed": passed,
        "global_check_count": int(len(checks)),
        "global_pass_count": int(checks["passed"].map(boolean_value).sum()),
        "configuration_error_count": int(len(configuration_errors)),
    }
    (output_root() / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise AssertionError(
            f"Final validation failed with {len(configuration_errors)} configuration errors"
        )


def validate_all() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    plan = json.loads(
        (COMPONENT_ROOT / "configs" / "run_plan.json").read_text(encoding="utf-8")
    )
    expected_by_configuration = configuration_split_seeds()
    target = int(plan["number_of_resplits_per_configuration"])
    configured = configured_configuration_keys()
    configured_counts = {
        f"{dataset}:{score}": len(seeds)
        for (dataset, score), seeds in expected_by_configuration.items()
    }
    add_check(
        rows,
        "configured_resplits_per_dataset_score",
        len(expected_by_configuration) == len(DATASETS) * len(SCORES)
        and all(
            len(seeds) == target and len(set(seeds)) == target
            for seeds in expected_by_configuration.values()
        ),
        json.dumps(configured_counts, sort_keys=True),
    )

    summary_dir = output_root() / "summaries"
    missing_outputs = [
        filename for filename in SUMMARY_FILES if not (summary_dir / filename).exists()
    ]
    add_check(
        rows,
        "required_summary_outputs_exist",
        not missing_outputs,
        "all present" if not missing_outputs else ";".join(missing_outputs),
    )
    per_split = pd.read_csv(summary_dir / "table2_all_scores_per_split_long.csv")
    cells = pd.read_csv(summary_dir / "table2_all_scores_cell_ledger_long.csv")
    groups = pd.read_csv(summary_dir / "table2_all_scores_group_ledger_long.csv")
    expected = target * len(DATASETS) * len(SCORES)
    failed = []
    for path in (output_root() / "splits").glob("seed_*/*/*__FAILED_*/FAILED.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            str(payload["dataset"]),
            str(payload["score"]),
            int(payload["split_seed"]),
        ) in configured:
            failed.append(path)
    add_check(
        rows,
        "completed_configuration_count",
        len(per_split) + len(failed) == expected,
        f"successful={len(per_split)}, failed={len(failed)}, expected={expected}",
    )
    add_check(
        rows,
        "completed_resplits_per_dataset_score",
        all(
            int(count) == target
            for count in per_split.groupby(["dataset", "score"]).size().tolist()
        )
        and len(per_split.groupby(["dataset", "score"])) == len(DATASETS) * len(SCORES),
        per_split.groupby(["dataset", "score"]).size().to_json(),
    )
    add_check(
        rows,
        "all_four_scores_run",
        set(per_split["score"].astype(str)) == set(SCORES),
        ",".join(sorted(set(per_split["score"].astype(str)))),
    )
    failures_preserved = all(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "failed"
        and bool(json.loads(path.read_text(encoding="utf-8")).get("traceback"))
        for path in failed
    )
    add_check(
        rows,
        "failed_configurations_preserved",
        failures_preserved,
        f"failed={len(failed)}; every failure retains a traceback",
    )

    validate_partitions(rows, per_split, expected_by_configuration)
    configuration_errors: list[str] = []
    for _, row in per_split.iterrows():
        configuration_errors.extend(validate_configuration(row, cells, groups))
    add_check(
        rows,
        "raw_ledger_reconstruction_all_configurations",
        not configuration_errors,
        f"errors={len(configuration_errors)}",
    )

    expected_done = target * len(DATASETS) * len(SCORES)
    done_count = len(
        [
            path
            for path in (output_root() / "splits").glob("seed_*/*/*/DONE.json")
            if (
                path.parents[1].name,
                path.parent.name,
                int(path.parents[2].name.removeprefix("seed_")),
            )
            in configured
        ]
    )
    add_check(
        rows,
        "success_and_failure_markers_account_for_target",
        done_count + len(failed) == expected_done,
        f"success={done_count}; failed={len(failed)}; expected={expected_done}",
    )
    add_check(
        rows,
        "negative_outcomes_retained",
        len(per_split) == done_count
        and set(per_split["heldout_improvement_category"].astype(str))
        <= {"positive", "tie", "negative"},
        per_split["heldout_improvement_category"].value_counts().to_json(),
    )
    add_check(
        rows,
        "optimizer_seed_fixed",
        (per_split["optimizer_seed"].astype(int) == 2026).all(),
        "optimizer_seed=2026 for every configuration",
    )
    checks = pd.DataFrame(rows)
    checks.to_csv(
        summary_dir / "table2_all_scores_validation_checks.csv", index=False
    )
    write_validation_report(checks, configuration_errors)
    return {
        "passed": True,
        "completed_configurations": int(len(per_split)),
        "failed_configurations": int(len(failed)),
        "configured_resplits_per_dataset_score": target,
        "configuration_errors": 0,
    }
