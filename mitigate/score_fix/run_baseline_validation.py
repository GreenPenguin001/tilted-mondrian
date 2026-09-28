from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(ROOT))

from atus.score_fix_atus import build_experiment as build_atus_experiment
from biobias.score_fix_biobias import build_experiment as build_biobias_experiment
from folktables_design_a.score_fix_folktables_design_a import build_experiment as build_folktables_a_experiment
from folktables_design_b.score_fix_folktables_design_b import build_experiment as build_folktables_b_experiment
from input_loading import LoadedDataset
from no_free_repair_core import ScoreFixConfig, validate_theta_zero_baselines
from runtime_paths import (
    DEFAULT_RESULT_ROOT,
    add_input_arguments,
    resolve_input_dirs,
)


EXPERIMENT_BUILDERS: dict[
    str, Callable[[Path | None], tuple[LoadedDataset, ScoreFixConfig]]
] = {
    "biobias": build_biobias_experiment,
    "folktables_design_a": build_folktables_a_experiment,
    "folktables_design_b": build_folktables_b_experiment,
    "atus": build_atus_experiment,
}


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("Baseline validation produced no rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["dataset"]), str(row["score_method"]), str(row["split"]))
        if key in seen:
            continue
        seen.add(key)
        summary_rows.append(row)

    lines = [
        "# Baseline G x Z Encoding Validation",
        "",
        "All rows were computed before any nonzero score-shift candidate was evaluated.",
        "The checks compare the frozen curve encodings, an independent group-Mondrian",
        "baseline calculation, and the generalized theta=0 repair path.",
        "",
        "| Dataset | Score | Split | Max GxZ gap | Mean GxZ gap | Avg set size | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {score_method} | {split} | {max_gap:.10f} | "
            "{mean_gap:.10f} | {set_size:.10f} | pass |".format(
                dataset=row["dataset"],
                score_method=row["score_method"],
                split=row["split"],
                max_gap=float(row["baseline_max_gz_gap"]),
                mean_gap=float(row["baseline_mean_gz_gap"]),
                set_size=float(row["baseline_average_set_size"]),
            )
        )
    lines.extend(
        [
            "",
            "Assertions passed:",
            "",
            "- Group IDs and ordering equal the frozen curve encoding.",
            "- Cell IDs, names, and ordering equal the frozen curve encoding.",
            "- Theta=0 covered indicators and set sizes equal an independent baseline calculation.",
            "- BioBias GxZ masks equal its GxY masks because Z=Y.",
            "",
            "The CSV contains every dataset, score method, split, G, and Z cell with row counts,",
            "covered counts, coverage, group covered counts, local-gap summaries, and set size.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate zero-shift baselines.")
    add_input_arguments(parser)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    arguments = parser.parse_args()
    input_dirs = resolve_input_dirs(arguments)
    output_dir = arguments.result_root / "baseline_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset, build_experiment in EXPERIMENT_BUILDERS.items():
        data, config = build_experiment(input_dirs[dataset])
        dataset_rows = validate_theta_zero_baselines(data, config)
        rows.extend(dataset_rows)
        print(f"baseline validation passed: {data.dataset} ({len(dataset_rows)} rows)")
    write_csv(rows, output_dir / "BASELINE_GZ_VALIDATION_REPORT.csv")
    write_markdown(rows, output_dir / "BASELINE_GZ_VALIDATION_REPORT.md")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
