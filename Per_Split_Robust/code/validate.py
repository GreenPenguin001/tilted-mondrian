"""Validate fixed-model random-resplit outputs from their run artifacts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .data import dataset_units, scores_for_unit
from .source_registry import COMPONENT_ROOT, output_root


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    detail: str = "",
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "detail": detail,
        }
    )


def validate_results() -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    plan = json.loads((COMPONENT_ROOT / "configs/run_plan.json").read_text(encoding="utf-8"))
    seed_config = json.loads(
        (COMPONENT_ROOT / "configs/split_seeds.json").read_text(encoding="utf-8")
    )
    expected_seeds = {int(seed) for seed in seed_config["active_seed_list"]}
    expected_count = int(plan["number_of_splits"])
    add_check(
        checks,
        "configured_seed_count",
        len(expected_seeds) == expected_count,
        len(expected_seeds),
        expected_count,
    )

    pool_manifests: dict[str, dict[str, Any]] = {}
    for unit in dataset_units():
        manifest_path = output_root() / "pools" / unit.key / "pool_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pool_manifests[unit.key] = manifest
        split_total = sum(
            int(manifest["canonical_split_sizes"][name])
            for name in ("cal", "repair", "test")
        )
        add_check(
            checks,
            f"pool_size:{unit.key}",
            split_total == int(manifest["n_pool"]),
            manifest["n_pool"],
            split_total,
        )
        add_check(
            checks,
            f"pool_train_zero:{unit.key}",
            int(manifest["train_rows"]) == 0,
            manifest["train_rows"],
            0,
        )
    done_paths = sorted((output_root() / "splits").glob("*/seed_*/DONE.json"), key=str)
    failure_paths = sorted((output_root() / "splits").glob("**/FAILED.json"), key=str)
    temporary_paths = sorted((output_root() / "splits").glob("**/.*.tmp_*"), key=str)
    expected_jobs = len(dataset_units()) * expected_count
    add_check(checks, "completed_job_count", len(done_paths) == expected_jobs, len(done_paths), expected_jobs)
    add_check(checks, "failed_job_count", not failure_paths, len(failure_paths), 0)
    add_check(checks, "temporary_directory_count", not temporary_paths, len(temporary_paths), 0)

    summary_frames: list[pd.DataFrame] = []
    artifact_failures: list[str] = []
    for done_path in done_paths:
        result_dir = done_path.parent
        done = json.loads(done_path.read_text(encoding="utf-8"))
        unit_key = result_dir.parent.name
        manifest = json.loads((result_dir / "split_manifest.json").read_text(encoding="utf-8"))
        config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
        with np.load(result_dir / "split_indices.npz") as indices, np.load(
            result_dir / "overlap_weights.npz"
        ) as weights:
            expected_sizes = pool_manifests[unit_key]["canonical_split_sizes"]
            combined = np.concatenate([indices[name] for name in ("cal", "repair", "test")])
            split_ok = (
                manifest["disjoint"] is True
                and manifest["union_is_full_pool"] is True
                and int(manifest["train_rows"]) == 0
                and all(
                    len(indices[name]) == int(expected_sizes[name])
                    for name in ("cal", "repair", "test")
                )
                and len(np.unique(combined)) == int(pool_manifests[unit_key]["n_pool"])
            )
            weight_lengths_ok = (
                len(weights["repair"]) == int(expected_sizes["repair"])
                and len(weights["test"]) == int(expected_sizes["test"])
            )
        config_ok = (
            config["fixed_model"] is True
            and config["base_classifier_retrained"] is False
            and int(config["train_rows_in_pool"]) == 0
            and config["propensity_refit_on_repair"] is True
            and config["test_used_for_selection"] is False
        )
        if not split_ok:
            artifact_failures.append(f"partition:{result_dir}")
        if not weight_lengths_ok:
            artifact_failures.append(f"weight_lengths:{result_dir}")
        if not config_ok:
            artifact_failures.append(f"run_config:{result_dir}")

        summary = pd.read_csv(result_dir / "summary.csv")
        summary_frames.append(summary)
        expected_scores = set(done["scores"])
        if set(summary["score"]) != expected_scores:
            artifact_failures.append(f"scores:{result_dir}")
        for score in expected_scores:
            score_dir = result_dir / score
            grid = pd.read_csv(score_dir / "cell_grid.csv")
            thresholds = pd.read_csv(score_dir / "thresholds.csv")
            methods_ok = set(grid["method"].unique()) == {"raw", "overlap"}
            support_ok = bool(
                grid.groupby(["alpha", "z"], dropna=False)["supported"].nunique().max()
                == 1
            )
            if not methods_ok or not support_ok or thresholds.empty:
                artifact_failures.append(f"score_grid:{score_dir}")

    add_check(
        checks,
        "partition_configuration_weight_and_grid_checks",
        not artifact_failures,
        len(artifact_failures),
        0,
        "; ".join(artifact_failures[:10]),
    )

    if summary_frames:
        job_long = pd.concat(summary_frames, ignore_index=True)
        counts = job_long.groupby(["dataset", "year", "score"], dropna=False)[
            "split_seed"
        ].nunique()
        expected_experiments = sum(len(scores_for_unit(unit)) for unit in dataset_units())
        add_check(
            checks,
            "experiment_row_count",
            len(counts) == expected_experiments,
            len(counts),
            expected_experiments,
        )
        add_check(
            checks,
            "seeds_per_experiment",
            bool((counts == expected_count).all()),
            f"min={counts.min()},max={counts.max()}",
            expected_count,
        )
        observed_seeds = set(job_long["split_seed"].astype(int).unique())
        add_check(
            checks,
            "configured_seed_list",
            observed_seeds == expected_seeds,
            sorted(observed_seeds),
            sorted(expected_seeds),
        )
        for column in (
            "same_test_rows_raw_overlap",
            "same_calibrated_sets_raw_overlap",
            "same_supported_universe_raw_overlap",
        ):
            add_check(checks, column, bool(job_long[column].all()), bool(job_long[column].all()), True)

    reported_path = output_root() / "summaries/table1_per_split_long.csv"
    reported_long = pd.read_csv(reported_path)
    diagnostics = reported_long["propensity_diagnostics"].map(json.loads)
    diagnostic_test_ess = diagnostics.map(lambda values: float(values["test_ess_fraction"]))
    ess_errors = (reported_long["ess_fraction"].astype(float) - diagnostic_test_ess).abs()
    add_check(
        checks,
        "ess_fraction_uses_test_split",
        bool((ess_errors <= 1e-12).all()),
        float(ess_errors.max()),
        "<=1e-12",
    )
    add_check(
        checks,
        "ess_reporting_split_is_test",
        bool(reported_long["ess_reporting_split"].eq("test").all()),
        sorted(reported_long["ess_reporting_split"].dropna().unique().tolist()),
        ["test"],
    )

    frame = pd.DataFrame(checks)
    output_dir = output_root() / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "table1_validation_checks.csv", index=False)
    passed = bool(frame["passed"].all())
    report = [
        "# Fixed-Model Resplit Validation",
        "",
        f"Overall status: **{'PASS' if passed else 'FAIL'}**",
        "",
        frame.to_markdown(index=False),
        "",
    ]
    (output_root() / "VALIDATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    status = {
        "passed": passed,
        "n_checks": len(frame),
        "n_failed_checks": int((~frame["passed"]).sum()),
        "n_completed_jobs": len(done_paths),
        "n_failed_jobs": len(failure_paths),
        "n_summary_rows": len(reported_long),
        "number_of_splits": expected_count,
    }
    status_path = output_root() / "validation.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Validation failed with {status['n_failed_checks']} failed checks")
    return frame


def main() -> None:
    validate_results()
    print("Fixed-model resplit validation passed.")


if __name__ == "__main__":
    main()
