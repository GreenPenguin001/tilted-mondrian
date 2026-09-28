from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "common"
sys.path.insert(0, str(COMMON))

from directional_asymmetric_core import (  # noqa: E402
    DirectionalArtifacts,
    SCORE_METHOD_ORDER,
    apply_directional_score_matrix,
    run_directional_asymmetric_experiment,
    write_frame,
)
from input_loading import LoadedDataset, SplitData  # noqa: E402
from runtime_paths import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    add_input_arguments,
    directional_result_dir,
    resolve_input_dirs,
    symmetric_result_dir,
)


DATASETS = (
    {
        "dataset": "biobias",
        "module_dir": "biobias",
        "module": "score_fix_biobias",
    },
    {
        "dataset": "folktables_design_a",
        "module_dir": "folktables_design_a",
        "module": "score_fix_folktables_design_a",
    },
    {
        "dataset": "folktables_design_b",
        "module_dir": "folktables_design_b",
        "module": "score_fix_folktables_design_b",
    },
    {
        "dataset": "atus",
        "module_dir": "atus",
        "module": "score_fix_atus",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tilted Mondrian intervention.")
    add_input_arguments(parser)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    arguments = parser.parse_args()
    input_dirs = resolve_input_dirs(arguments)
    aggregate_dir = arguments.result_root / "directional_asymmetric"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    stage_start = time.perf_counter()
    aggregates: dict[str, list[dict[str, Any]]] = {
        "summary": [],
        "offset": [],
        "cell": [],
        "group": [],
        "test": [],
        "search": [],
        "validation": [],
    }

    for specification in DATASETS:
        module_dir = ROOT / specification["module_dir"]
        sys.path.insert(0, str(module_dir))
        module = importlib.import_module(specification["module"])
        data, config = module.build_experiment(input_dirs[specification["dataset"]])
        artifacts = run_directional_asymmetric_experiment(
            data=data,
            score_config=config,
            result_dir=directional_result_dir(arguments.result_root, specification["dataset"]),
            symmetric_result_dir=symmetric_result_dir(
                arguments.result_root, specification["dataset"]
            ),
        )
        collect_artifacts(aggregates, artifacts)
        for row in artifacts.summary_rows:
            print(
                f"{row['dataset']} {row['score_method']}: "
                f"repair max={row['directional_repair_max_gap']:.12g}, "
                f"test max={row['directional_test_max_gap']:.12g}, "
                f"candidates={row['candidate_count_total']}, "
                f"runtime={row['runtime_seconds']:.3f}s",
                flush=True,
            )

    aggregates["validation"].extend(intervention_mode_unit_checks())
    if not all(bool(row["passed"]) for row in aggregates["validation"]):
        failures = [
            row
            for row in aggregates["validation"]
            if not bool(row["passed"])
        ]
        raise AssertionError(
            f"Tilted Mondrian validation failures: {failures}"
        )

    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_ALL_SUMMARIES.csv",
        aggregates["summary"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_OFFSETS.csv",
        aggregates["offset"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_CELL_LEDGER.csv",
        aggregates["cell"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_GROUP_LEDGER.csv",
        aggregates["group"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_TEST_RESULTS.csv",
        aggregates["test"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_SEARCH_DIAGNOSTICS.csv",
        aggregates["search"],
    )
    write_frame(
        aggregate_dir / "DIRECTIONAL_ASYMMETRIC_VALIDATION_REPORT.csv",
        aggregates["validation"],
    )
    write_validation_markdown(aggregates["validation"], aggregate_dir)
    write_algorithm_notes(aggregate_dir)
    elapsed = float(time.perf_counter() - stage_start)
    (aggregate_dir / "DIRECTIONAL_ASYMMETRIC_RUNTIME.json").write_text(
        json.dumps(
            {
                "total_runtime_seconds": elapsed,
                "experiment_count": len(aggregates["summary"]),
                "score_order": list(SCORE_METHOD_ORDER),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"Completed 16 Tilted Mondrian experiments in "
        f"{elapsed:.3f} seconds.",
        flush=True,
    )


def collect_artifacts(
    aggregates: dict[str, list[dict[str, Any]]],
    artifacts: DirectionalArtifacts,
) -> None:
    aggregates["summary"].extend(artifacts.summary_rows)
    aggregates["offset"].extend(artifacts.offset_rows)
    aggregates["cell"].extend(artifacts.cell_rows)
    aggregates["group"].extend(artifacts.group_rows)
    aggregates["test"].extend(artifacts.test_rows)
    aggregates["search"].extend(artifacts.search_rows)
    aggregates["validation"].extend(artifacts.validation_rows)


def intervention_mode_unit_checks() -> list[dict[str, Any]]:
    base = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    split = SplitData(
        probs=np.full((2, 2), 0.5),
        y=np.asarray([0, 1]),
        g=np.asarray(["a", "b"], dtype=object),
        z=np.asarray(["0", "1"], dtype=object),
    )
    common = {
        "dataset": "synthetic",
        "splits": {
            "repair": split,
            "calibration": split,
            "test": split,
        },
        "label_values": ["0", "1"],
        "label_names": {"0": "zero", "1": "one"},
        "group_values": ["a", "b"],
        "group_names": {"a": "A", "b": "B"},
        "cell_values": ["0", "1"],
        "cell_names": {"0": "zero", "1": "one"},
        "source_format": "synthetic",
        "metadata": {},
    }
    roles = {
        0: {
            "repairable": True,
            "advantaged_group": "a",
            "disadvantaged_group": "b",
        },
        1: {
            "repairable": True,
            "advantaged_group": "b",
            "disadvantaged_group": "a",
        },
    }
    plus = np.asarray([0.01, 0.02])
    minus = np.asarray([0.03, 0.04])
    label_data = LoadedDataset(
        intervention_mode="label_defined_z", **common
    )
    label_actual = apply_directional_score_matrix(
        base, split, label_data, plus, minus, roles
    )
    label_expected = np.asarray([[0.11, 0.16], [0.27, 0.42]])
    covariate_data = LoadedDataset(
        intervention_mode="covariate_defined_z", **common
    )
    covariate_actual = apply_directional_score_matrix(
        base, split, covariate_data, plus, minus, roles
    )
    covariate_expected = np.asarray([[0.11, 0.21], [0.32, 0.42]])
    return [
        {
            "dataset": "synthetic",
            "score_method": "all",
            "check": "label_defined_directional_shifting_is_correct",
            "passed": bool(
                np.allclose(
                    label_actual,
                    label_expected,
                    rtol=0.0,
                    atol=1e-15,
                )
            ),
            "detail": "candidate label receives its cell-specific fixed-direction magnitude",
        },
        {
            "dataset": "synthetic",
            "score_method": "all",
            "check": "covariate_defined_directional_shifting_is_correct",
            "passed": bool(
                np.allclose(
                    covariate_actual,
                    covariate_expected,
                    rtol=0.0,
                    atol=1e-15,
                )
            ),
            "detail": "row G x Z shift is shared across all candidate labels",
        },
    ]


def write_validation_markdown(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(rows)
    scientific = frame.loc[frame["dataset"] != "all"]
    grouped = (
        scientific.groupby(["dataset", "score_method"])["passed"]
        .all()
        .reset_index()
    )
    text = f"""# Tilted Mondrian Validation Report

All executable checks passed. Zero shifts reproduce the baseline, both intervention modes are correct, magnitudes are nonnegative, the selected symmetric solution is included and reproduced, repair worst gap weakly dominates it, all four repair constraints hold, calibration group counts are exact, and test is excluded from search.

{grouped.to_markdown(index=False)}
"""
    (output_dir / "DIRECTIONAL_ASYMMETRIC_VALIDATION_REPORT.md").write_text(
        text, encoding="utf-8"
    )


def write_algorithm_notes(output_dir: Path) -> None:
    text = """# Tilted Mondrian

## Intervention parameterization

The symmetric intervention uses one nonnegative magnitude per Z cell. It raises the initially advantaged group's score and lowers the initially disadvantaged group's score by the same amount. Tilted Mondrian relaxes only that equal-magnitude coupling: `theta_plus[z]` controls the advantaged-side score increase and `theta_minus[z]` controls the disadvantaged-side score decrease. Both remain nonnegative, so the scientific directions remain fixed and arbitrary-sign group-cell offsets are not allowed.

## Constraints and data protocol

The full repair split fixes roles, selects the intervention, and enforces exact baseline group covered counts, the 0.01 cell-wise advantaged loss cap, disadvantaged-cell non-harm, and local-gap nonincrease. No constraint becomes a penalty. The intervention is frozen before calibration; calibration only reselects group-Mondrian thresholds to match baseline calibration group counts. Test is held out and is never supplied to search or candidate ranking.

## Search framework

The method includes the zero candidate, simultaneous line search, feasible-boundary projection, critical score-event grids, coordinate refinement, single/pair/selected-triple cell blocks, beam width, deterministic Halton candidates, seed, tolerances, and candidate controls. Each cell is searched as the two-dimensional block `(theta_plus[z], theta_minus[z])`, including advantaged-only, disadvantaged-only, joint asymmetric, and symmetric-subspace moves. Prediction-set size is the final lexicographic tie-breaker.

The selected symmetric solution is explicitly re-evaluated as `theta_plus = theta_minus = theta_symmetric`, retained during beam pruning, and included in the final feasible pool. Therefore Tilted Mondrian cannot be worse than the symmetric intervention on the repair-split worst-gap objective unless implementation or ranking is incorrect.

## Interpretation

Repair-split improvement does not imply held-out improvement because roles and magnitudes are selected on the repair sample and test is evaluated only after selection.
"""
    (output_dir / "DIRECTIONAL_ASYMMETRIC_ALGORITHM_NOTES.md").write_text(
        text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
