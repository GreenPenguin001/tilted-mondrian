"""Ledger-driven aggregation for Table 2 robustness outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .source_registry import (
    DATASETS,
    SCORES,
    configured_configuration_keys,
    output_root,
)


def summary_dir() -> Path:
    return output_root() / "summaries"
METRIC_FIELDS = (
    "baseline_test_max_gap",
    "directional_test_max_gap",
    "directional_test_delta_max_gap",
    "heldout_maximum_single_cell_mitigation",
    "maximum_advantaged_harm_repair",
    "test_average_set_size_change",
    "baseline_repair_max_gap",
    "directional_repair_max_gap",
    "repair_total_mitigation",
    "repair_certificate",
    "total_runtime_seconds",
)


def completed_configuration_dirs() -> list[Path]:
    configured = configured_configuration_keys()
    paths = [
        path.parent for path in (output_root() / "splits").glob("seed_*/*/*/DONE.json")
    ]
    return sorted(
        [
            path
            for path in paths
            if (
                path.parent.name,
                path.name,
                int(path.parents[1].name.removeprefix("seed_")),
            )
            in configured
        ],
        key=str,
    )


def load_per_split() -> pd.DataFrame:
    rows = [pd.read_csv(path / "summary.csv") for path in completed_configuration_dirs()]
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    dataset_order = [specification.dataset for specification in DATASETS]
    output["dataset"] = pd.Categorical(
        output["dataset"], categories=dataset_order, ordered=True
    )
    output["score"] = pd.Categorical(
        output["score"], categories=list(SCORES), ordered=True
    )
    output = output.sort_values(["split_seed", "dataset", "score"]).reset_index(drop=True)
    output["dataset"] = output["dataset"].astype(str)
    output["score"] = output["score"].astype(str)
    return output


def enriched_cell_ledger(configuration: Path, summary: pd.Series) -> pd.DataFrame:
    ledger = pd.read_csv(
        configuration / "directional" / "directional_asymmetric_cell_ledger.csv"
    )
    roles = json.loads(summary["repair_defined_roles"])
    ledger["split_seed"] = int(summary["split_seed"])
    ledger["score"] = summary["score"]
    ledger["cell_id"] = ledger["z"].astype(str)
    ledger["cell_name"] = ledger["z_name"].astype(str)
    ledger["support_count"] = ledger["n"].astype(int)
    ledger["repairable"] = ledger["cell_id"].map(
        lambda value: bool(roles[str(value)]["repairable"])
    )
    ledger["repair_defined_role"] = ledger["role"]
    ledger["directional_coverage"] = ledger["repaired_coverage"]
    baseline_gaps = {}
    directional_gaps = {}
    for (split, cell), rows in ledger.groupby(["split", "cell_id"], sort=False):
        values = rows["baseline_coverage"].astype(float).tolist()
        repaired = rows["repaired_coverage"].astype(float).tolist()
        baseline_gaps[(split, cell)] = abs(values[0] - values[1]) if len(values) == 2 else np.nan
        directional_gaps[(split, cell)] = abs(repaired[0] - repaired[1]) if len(repaired) == 2 else np.nan
    ledger["baseline_absolute_local_gap"] = [
        baseline_gaps[(row.split, row.cell_id)] for row in ledger.itertuples()
    ]
    ledger["directional_absolute_local_gap"] = [
        directional_gaps[(row.split, row.cell_id)] for row in ledger.itertuples()
    ]
    ledger["local_gap_mitigation"] = (
        ledger["baseline_absolute_local_gap"] - ledger["directional_absolute_local_gap"]
    )
    return ledger


def load_ledgers(per_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = []
    groups = []
    summary_lookup = {
        (int(row.split_seed), str(row.dataset), str(row.score)): row
        for row in per_split.itertuples()
    }
    for configuration in completed_configuration_dirs():
        score = configuration.name
        dataset = configuration.parent.name
        split_seed = int(configuration.parents[1].name.removeprefix("seed_"))
        summary = pd.Series(summary_lookup[(split_seed, dataset, score)]._asdict())
        cells.append(enriched_cell_ledger(configuration, summary))
        group = pd.read_csv(
            configuration / "directional" / "directional_asymmetric_group_ledger.csv"
        )
        group["split_seed"] = split_seed
        group["score"] = score
        groups.append(group)
    return pd.concat(cells, ignore_index=True), pd.concat(groups, ignore_index=True)


def empirical_stats(values: pd.Series) -> dict[str, float | int]:
    array = values.astype(float).to_numpy()
    return {
        "n_successful_splits": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "p5": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def split_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, score), group in per_split.groupby(["dataset", "score"], sort=False):
        base = {
            "dataset": dataset,
            "score": score,
            "n_successful_splits": int(len(group)),
            "positive_delta_frequency": float((group["heldout_improvement_category"] == "positive").mean()),
            "tie_delta_frequency": float((group["heldout_improvement_category"] == "tie").mean()),
            "negative_delta_frequency": float((group["heldout_improvement_category"] == "negative").mean()),
            "repair_constraint_pass_rate": float(group["every_repair_constraint_pass"].astype(bool).mean()),
            "directional_better_than_symmetric_frequency": float((group["directional_test_max_gap"] < group["symmetric_test_max_gap"] - 1e-10).mean()),
            "directional_equal_to_symmetric_frequency": float((abs(group["directional_test_max_gap"] - group["symmetric_test_max_gap"]) <= 1e-10).mean()),
            "directional_worse_than_symmetric_frequency": float((group["directional_test_max_gap"] > group["symmetric_test_max_gap"] + 1e-10).mean()),
            "zero_intervention_frequency": float(group["zero_intervention_selected_flag"].astype(bool).mean()),
        }
        for metric in METRIC_FIELDS:
            stats = empirical_stats(group[metric])
            for name, value in stats.items():
                if name == "n_successful_splits":
                    continue
                base[f"{metric}_{name}"] = value
        rows.append(base)
    return pd.DataFrame(rows)


def grouped_summary(per_split: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, group in per_split.groupby(key, sort=False):
        row: dict[str, Any] = {key: value, "n_configurations": int(len(group))}
        for metric in (
            "directional_test_delta_max_gap",
            "heldout_maximum_single_cell_mitigation",
            "maximum_advantaged_harm_repair",
            "test_average_set_size_change",
        ):
            stats = empirical_stats(group[metric])
            row.update({f"{metric}_{name}": stat for name, stat in stats.items()})
        row["positive_frequency"] = float((group["heldout_improvement_category"] == "positive").mean())
        row["negative_frequency"] = float((group["heldout_improvement_category"] == "negative").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def constraint_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "repair_group_covered_counts_exact",
        "calibration_group_covered_counts_exact",
        "every_repair_constraint_pass",
        "symmetric_candidate_flag",
        "symmetric_candidate_equivalence_flag",
        "directional_repair_not_worse_than_symmetric_flag",
    ]
    rows = []
    for (dataset, score), group in per_split.groupby(["dataset", "score"], sort=False):
        row = {"dataset": dataset, "score": score, "n": len(group)}
        row.update({f"{field}_rate": float(group[field].astype(bool).mean()) for field in fields})
        row["max_advantaged_harm"] = float(group["maximum_advantaged_harm_repair"].max())
        row["min_disadvantaged_change"] = float(group["minimum_disadvantaged_cell_change_repair"].min())
        row["max_local_gap_increase"] = float(group["maximum_local_gap_increase_repair"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def role_change_summary(cells: pd.DataFrame) -> pd.DataFrame:
    repair = cells.loc[cells["split"] == "repair"].copy()
    repair = repair.loc[repair["repair_defined_role"].isin(["advantaged", "disadvantaged"])]
    rows = []
    for (dataset, score, cell_id, cell_name), group in repair.groupby(
        ["dataset", "score", "cell_id", "cell_name"], sort=False
    ):
        adv = group.loc[group["repair_defined_role"] == "advantaged"]
        frequencies = adv["group"].astype(str).value_counts(normalize=True).to_dict()
        rows.append(
            {
                "dataset": dataset,
                "score": score,
                "cell_id": cell_id,
                "cell_name": cell_name,
                "n_splits": int(adv["split_seed"].nunique()),
                "advantaged_group_frequencies": json.dumps(frequencies, sort_keys=True),
                "role_change_frequency": float(1.0 - max(frequencies.values())) if frequencies else 0.0,
            }
        )
    return pd.DataFrame(rows)


def failure_summary() -> pd.DataFrame:
    rows = []
    configured = configured_configuration_keys()
    for path in (output_root() / "splits").glob("seed_*/*/*__FAILED_*/FAILED.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(payload["dataset"]),
            str(payload["score"]),
            int(payload["split_seed"]),
        )
        if key not in configured:
            continue
        rows.append({"path": str(path), **payload})
    columns = [
        "path",
        "status",
        "dataset",
        "score",
        "split_seed",
        "failure_stage",
        "error_type",
        "error_message",
        "traceback",
        "exact_command",
        "runtime_seconds",
    ]
    return pd.DataFrame(rows, columns=columns)


def appendix_ready(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset"],
                "score": str(row["score"]).upper(),
                "N": row["n_successful_splits"],
                "heldout_baseline_Lmax_mean_sd": f"{row['baseline_test_max_gap_mean']:.4f} $\\pm$ {row['baseline_test_max_gap_sample_sd']:.4f}",
                "heldout_directional_Lmax_mean_sd": f"{row['directional_test_max_gap_mean']:.4f} $\\pm$ {row['directional_test_max_gap_sample_sd']:.4f}",
                "heldout_delta_Lmax_mean_sd": f"{row['directional_test_delta_max_gap_mean']:.4f} $\\pm$ {row['directional_test_delta_max_gap_sample_sd']:.4f}",
                "heldout_max_cell_mitigation_mean_sd": f"{row['heldout_maximum_single_cell_mitigation_mean']:.4f} $\\pm$ {row['heldout_maximum_single_cell_mitigation_sample_sd']:.4f}",
                "repair_harm_mean_sd": f"{row['maximum_advantaged_harm_repair_mean']:.4f} $\\pm$ {row['maximum_advantaged_harm_repair_sample_sd']:.4f}",
                "heldout_set_size_change_mean_sd": f"{row['test_average_set_size_change_mean']:.4f} $\\pm$ {row['test_average_set_size_change_sample_sd']:.4f}",
                "positive_tie_negative_frequency": f"{row['positive_delta_frequency']:.2f}/{row['tie_delta_frequency']:.2f}/{row['negative_delta_frequency']:.2f}",
                "repair_feasibility_rate": f"{row['repair_constraint_pass_rate']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def write_appendix_tex(table: pd.DataFrame) -> None:
    path = summary_dir() / "table2_all_scores_appendix_ready.tex"
    lines = [
        "% Fixed-model random-resplit empirical variability; not independent model-training runs or independent-data confidence intervals.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrcccccccc}",
        "\\toprule",
        "Dataset & Score & $N$ & Test $L^0_{\\max}$ & Test $L^\\theta_{\\max}$ & Test $\\Delta L_{\\max}$ & Test $\\max_z m_z$ & Repair Harm & Test $\\Delta|C|$ & Pos/Tie/Neg & Feasible \\\\",
        "\\midrule",
    ]
    for row in table.to_dict("records"):
        dataset = str(row["dataset"]).replace("_", "\\_")
        lines.append(
            f"{dataset} & {row['score']} & {row['N']} & "
            f"{row['heldout_baseline_Lmax_mean_sd']} & {row['heldout_directional_Lmax_mean_sd']} & "
            f"{row['heldout_delta_Lmax_mean_sd']} & {row['heldout_max_cell_mitigation_mean_sd']} & "
            f"{row['repair_harm_mean_sd']} & {row['heldout_set_size_change_mean_sd']} & "
            f"{row['positive_tie_negative_frequency']} & {row['repair_feasibility_rate']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Fixed-model random-resplit empirical variability. $L_{\\max}$, $\\Delta L_{\\max}$, $\\max_z m_z$, and $\\Delta|C|$ are held-out test diagnostics. Harm and feasibility are repair-split quantities. Cells, support, and advantaged/disadvantaged roles are defined on each baseline repair split and frozen. The predictor is fixed; repair and calibration are rerun per split. Negative held-out changes are retained. These are not independent model-training runs or independent-data confidence intervals.}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_robustness(per_split: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    datasets = [specification.dataset for specification in DATASETS]
    combinations = [(dataset, score) for dataset in datasets for score in SCORES]
    labels = [f"{dataset.replace('folktables_design_', 'Folk-').replace('biobias', 'BioBias').replace('atus', 'ATUS')}\n{score.upper()}" for dataset, score in combinations]
    colors = {"simple": "#0072B2", "aps": "#E69F00", "raps": "#009E73", "saps": "#CC79A7"}
    panels = [
        ("baseline_test_max_gap", "directional_test_max_gap", "Held-out worst local gap", None),
        (None, "directional_test_delta_max_gap", "Held-out worst-gap reduction", 0.0),
        (None, "heldout_maximum_single_cell_mitigation", "Held-out maximum cell mitigation", 0.0),
        (None, "maximum_advantaged_harm_repair", "Repair advantaged-cell harm", 0.01),
        (None, "test_average_set_size_change", "Held-out average set-size change", 0.0),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    rng = np.random.default_rng(2026)
    for axis, (baseline_field, field, title, reference) in zip(axes.flat, panels):
        for index, (dataset, score) in enumerate(combinations):
            values = per_split.loc[
                (per_split["dataset"] == dataset) & (per_split["score"] == score), field
            ].astype(float)
            jitter = rng.uniform(-0.16, 0.16, len(values))
            axis.scatter(index + jitter, values, s=14, alpha=0.62, color=colors[score], linewidths=0)
            if baseline_field:
                baseline = per_split.loc[
                    (per_split["dataset"] == dataset) & (per_split["score"] == score), baseline_field
                ].astype(float)
                axis.scatter(
                    np.full(len(baseline), index + 0.22, dtype=float),
                    baseline,
                    s=9,
                    alpha=0.25,
                    color="#666666",
                    linewidths=0,
                )
        if reference is not None:
            axis.axhline(reference, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    fig.suptitle("Table 2 fixed-model random-resplit robustness", fontsize=15)
    fig.savefig(summary_dir() / "table2_all_scores_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_all() -> dict[str, int]:
    output_dir = summary_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_split = load_per_split()
    if per_split.empty:
        raise RuntimeError("No completed configurations to aggregate")
    cells, groups = load_ledgers(per_split)
    summary = split_summary(per_split)
    dataset_summary = grouped_summary(per_split, "dataset")
    score_summary = grouped_summary(per_split, "score")
    constraints = constraint_summary(per_split)
    roles = role_change_summary(cells)
    failures = failure_summary()
    appendix = appendix_ready(summary)
    runtime = (
        per_split.groupby(["dataset", "score"], sort=False)[
            ["symmetric_runtime_seconds", "directional_runtime_seconds", "total_runtime_seconds"]
        ]
        .agg(["mean", "sum"])
        .reset_index()
    )
    runtime.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in runtime.columns
    ]
    outputs = {
        "table2_all_scores_per_split_long.csv": per_split,
        "table2_all_scores_split_summary.csv": summary,
        "table2_all_scores_dataset_summary.csv": dataset_summary,
        "table2_all_scores_score_summary.csv": score_summary,
        "table2_all_scores_constraint_summary.csv": constraints,
        "table2_all_scores_cell_ledger_long.csv": cells,
        "table2_all_scores_group_ledger_long.csv": groups,
        "table2_all_scores_role_change_summary.csv": roles,
        "table2_all_scores_runtime_summary.csv": runtime,
        "table2_all_scores_failure_summary.csv": failures,
        "table2_all_scores_appendix_ready.csv": appendix,
    }
    for filename, output in outputs.items():
        output.to_csv(output_dir / filename, index=False)
    write_appendix_tex(appendix)
    plot_robustness(per_split)
    return {
        "completed_configurations": int(len(per_split)),
        "completed_seeds": int(per_split["split_seed"].nunique()),
        "failed_configurations": int(len(failures)),
    }
