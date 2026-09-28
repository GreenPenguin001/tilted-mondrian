from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "common"
sys.path.insert(0, str(COMMON))

from runtime_paths import DEFAULT_RESULT_ROOT, directional_result_dir  # noqa: E402

SCORE_ORDER = ("simple", "aps", "raps", "saps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Tilted Mondrian outputs.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    arguments = parser.parse_args()
    results = {
        dataset: directional_result_dir(arguments.result_root, dataset)
        for dataset in (
            "biobias",
            "folktables_design_a",
            "folktables_design_b",
            "atus",
        )
    }
    required = {
        "directional_asymmetric_summary.csv",
        "directional_asymmetric_offsets.csv",
        "directional_asymmetric_cell_ledger.csv",
        "directional_asymmetric_group_ledger.csv",
        "directional_asymmetric_test_results.csv",
        "directional_asymmetric_search_diagnostics.csv",
        "directional_asymmetric_validation.csv",
        "config.json",
        "README_RESULT.md",
    }
    for dataset, result_dir in results.items():
        missing = sorted(
            name for name in required if not (result_dir / name).exists()
        )
        if missing:
            raise FileNotFoundError(
                f"{dataset}: missing directional result files: {missing}"
            )
        summary = pd.read_csv(
            result_dir / "directional_asymmetric_summary.csv"
        )
        if tuple(summary["score_method"].astype(str)) != SCORE_ORDER:
            raise AssertionError(f"{dataset}: score order changed")
        if not (
            summary["maximum_advantaged_harm_repair"].astype(float)
            <= 0.01 + 1e-9
        ).all():
            raise AssertionError(
                f"{dataset}: advantaged harm constraint failed"
            )
        if not (
            summary[
                "minimum_disadvantaged_cell_change_repair"
            ].astype(float)
            >= -1e-9
        ).all():
            raise AssertionError(
                f"{dataset}: disadvantaged non-harm failed"
            )
        if not (
            summary["maximum_local_gap_increase_repair"].astype(
                float
            )
            <= 1e-9
        ).all():
            raise AssertionError(
                f"{dataset}: local gap nonincrease failed"
            )
        if not summary[
            "repair_group_covered_counts_exact"
        ].astype(bool).all():
            raise AssertionError(
                f"{dataset}: repair group counts changed"
            )
        if not summary[
            "calibration_group_covered_counts_exact"
        ].astype(bool).all():
            raise AssertionError(
                f"{dataset}: calibration group counts changed"
            )
        if not (
            summary["directional_repair_max_gap"].astype(float)
            <= summary["symmetric_repair_max_gap"].astype(float) + 1e-8
        ).all():
            raise AssertionError(
                f"{dataset}: symmetric containment failed"
            )
        if not summary["symmetric_candidate_in_pool"].astype(bool).all():
            raise AssertionError(
                f"{dataset}: symmetric candidate was not retained"
            )
        offsets = pd.read_csv(
            result_dir / "directional_asymmetric_offsets.csv"
        )
        if not (
            (offsets["theta_plus"].astype(float) >= 0.0)
            & (offsets["theta_minus"].astype(float) >= 0.0)
        ).all():
            raise AssertionError(
                f"{dataset}: negative directional magnitude"
            )
        groups = pd.read_csv(
            result_dir / "directional_asymmetric_group_ledger.csv"
        )
        for split_name in ("repair", "calibration"):
            rows = groups.loc[groups["split"] == split_name]
            if not (
                rows["covered_count_change"].astype(int) == 0
            ).all():
                raise AssertionError(
                    f"{dataset}: {split_name} group count changed"
                )
        diagnostics = pd.read_csv(
            result_dir
            / "directional_asymmetric_search_diagnostics.csv",
            low_memory=False,
        )
        if not (
            diagnostics["optimizer_phase"] == "symmetric_candidate"
        ).any():
            raise AssertionError(
                f"{dataset}: no symmetric candidate trace row"
            )

    aggregate_dir = arguments.result_root / "directional_asymmetric"
    validation = pd.read_csv(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_VALIDATION_REPORT.csv"
    )
    if not validation["passed"].astype(bool).all():
        failures = validation.loc[
            ~validation["passed"].astype(bool)
        ]
        raise AssertionError(
            f"Global Tilted Mondrian validation failed:\n"
            f"{failures.to_string(index=False)}"
        )
    aggregate_summary = pd.read_csv(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_ALL_SUMMARIES.csv"
    )
    if not (
        aggregate_summary["directional_repair_max_gap"].astype(float)
        <= aggregate_summary["symmetric_repair_max_gap"].astype(float) + 1e-8
    ).all():
        raise AssertionError(
            "Tilted Mondrian failed to contain the symmetric solution"
        )
    assert_human_facing_gender_terminology(arguments.result_root)
    print(
        "All Tilted Mondrian validations passed for 16 experiments."
    )


def assert_human_facing_gender_terminology(result_root: Path) -> None:
    report_paths = [
        path
        for path in Path(result_root).rglob("*")
        if path.is_file()
        and (
            path.name.startswith("DIRECTIONAL_ASYMMETRIC")
            or any(
                part.startswith("directional_asymmetric_result_")
                for part in path.parts
            )
        )
        and path.suffix.lower() in {".csv", ".md", ".json"}
    ]
    forbidden = re.compile(r"\bsex\b", flags=re.IGNORECASE)
    for path in report_paths:
        text = path.read_text(encoding="utf-8")
        if forbidden.search(text):
            raise AssertionError(
                f"Human-facing directional output uses forbidden "
                f"terminology: {path}"
            )


if __name__ == "__main__":
    main()
