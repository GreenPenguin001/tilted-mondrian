"""Aggregate completed resplits into descriptive Table 1 robustness outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .source_registry import COMPONENT_ROOT, output_root


SUMMARY_METRICS = (
    "raw_reversal_count",
    "overlap_reversal_count",
    "raw_supported_conflict_strength",
    "overlap_supported_conflict_strength",
    "paired_absolute_conflict_reduction",
    "paired_relative_conflict_reduction",
    "raw_high_conflict_count",
    "overlap_high_conflict_count",
    "ess_fraction",
    "average_set_size_change",
)


def load_run_plan() -> dict[str, Any]:
    return json.loads((COMPONENT_ROOT / "configs/run_plan.json").read_text(encoding="utf-8"))


def completed_summary_files() -> list[Path]:
    paths = []
    for done_path in (output_root() / "splits").glob("*/seed_*/DONE.json"):
        summary_path = done_path.parent / "summary.csv"
        if summary_path.exists():
            paths.append(summary_path)
    return sorted(paths, key=str)


def apply_test_ess_reporting(frame: pd.DataFrame) -> pd.DataFrame:
    """Report ESS from the held-out test split saved in propensity diagnostics."""
    corrected = frame.copy()
    diagnostics = corrected["propensity_diagnostics"].map(json.loads)
    missing = [index for index, values in diagnostics.items() if "test_ess_fraction" not in values]
    if missing:
        raise ValueError(f"Missing test_ess_fraction in propensity diagnostics for rows: {missing[:10]}")
    corrected["ess_fraction"] = diagnostics.map(lambda values: float(values["test_ess_fraction"]))
    corrected["ess_reporting_split"] = "test"
    return corrected


def load_long_results() -> pd.DataFrame:
    paths = completed_summary_files()
    if not paths:
        return pd.DataFrame()
    long = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    long = apply_test_ess_reporting(long)
    return long.sort_values(["dataset", "year", "score", "split_seed"], na_position="first").reset_index(drop=True)


def descriptive_statistics(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "year", "score"]
    for keys, subset in long.groupby(group_columns, dropna=False, sort=True):
        dataset, year, score = keys
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "year": year,
                    "score": score,
                    "metric": metric,
                    "n_successful_splits": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p95": float(np.quantile(values, 0.95)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def frequency_summary(long: pd.DataFrame) -> pd.DataFrame:
    plan = load_run_plan()
    thresholds = plan["adequate_ess_fraction_thresholds"]
    expected = int(plan["number_of_splits"])
    rows = []
    for keys, subset in long.groupby(["dataset", "year", "score"], dropna=False, sort=True):
        dataset, year, score = keys
        n = len(subset)
        rows.append(
            {
                "dataset": dataset,
                "year": year,
                "score": score,
                "n_successful_splits": n,
                "n_expected_splits": expected,
                "n_failed_splits": expected - n,
                "failure_rate": float((expected - n) / expected),
                "frequency_positive_conflict_strength_reduction": float((subset["paired_absolute_conflict_reduction"] > 0).mean()),
                "frequency_reversal_count_reduction": float((subset["raw_reversal_count"] > subset["overlap_reversal_count"]).mean()),
                "frequency_reversal_count_increase": float((subset["raw_reversal_count"] < subset["overlap_reversal_count"]).mean()),
                "adequate_ess_threshold": float(thresholds[dataset]),
                "frequency_adequate_ess": float((subset["ess_fraction"] >= float(thresholds[dataset])).mean()),
            }
        )
    return pd.DataFrame(rows)


def appendix_ready(long: pd.DataFrame, stats: pd.DataFrame, frequencies: pd.DataFrame) -> pd.DataFrame:
    keyed_stats = stats.assign(year_key=stats["year"].fillna(-1).astype(int))
    keyed_frequencies = frequencies.assign(year_key=frequencies["year"].fillna(-1).astype(int)).drop(
        columns="year"
    )
    metrics = keyed_stats.pivot_table(
        index=["dataset", "year_key", "score"], columns="metric", values=["mean", "std", "median", "q1", "q3", "p05", "p95", "minimum", "maximum"]
    )
    metrics.columns = [f"{metric}_{stat}" for stat, metric in metrics.columns]
    metrics = metrics.reset_index()
    out = metrics.merge(keyed_frequencies, on=["dataset", "year_key", "score"], how="left")
    out["year"] = out["year_key"].replace(-1, np.nan)
    out = out.drop(columns="year_key")
    return out.sort_values(["dataset", "year", "score"], na_position="first")


def write_latex_table(appendix: pd.DataFrame, path: Path) -> None:
    rows = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Dataset & Year & Score & Raw strength & Overlap strength & Reduction & ESS fraction \\",
        r"\midrule",
    ]
    for row in appendix.itertuples(index=False):
        year = "--" if pd.isna(row.year) else str(int(row.year))
        raw = f"{getattr(row, 'raw_supported_conflict_strength_mean'):.4f} $\\pm$ {getattr(row, 'raw_supported_conflict_strength_std'):.4f}"
        overlap = f"{getattr(row, 'overlap_supported_conflict_strength_mean'):.4f} $\\pm$ {getattr(row, 'overlap_supported_conflict_strength_std'):.4f}"
        reduction = f"{getattr(row, 'paired_absolute_conflict_reduction_mean'):.4f} $\\pm$ {getattr(row, 'paired_absolute_conflict_reduction_std'):.4f}"
        ess = f"{getattr(row, 'ess_fraction_mean'):.3f} $\\pm$ {getattr(row, 'ess_fraction_std'):.3f}"
        dataset = str(row.dataset).replace("_", r"\_")
        rows.append(f"{dataset} & {year} & {row.score.upper()} & {raw} & {overlap} & {reduction} & {ess} \\\\")
    rows.extend((r"\bottomrule", r"\end{tabular}", ""))
    path.write_text("\n".join(rows), encoding="utf-8")


def make_robustness_figure(long: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (
        long[["dataset", "year", "score"]]
        .drop_duplicates()
        .assign(
            label=lambda frame: frame.apply(
                lambda row: f"{row.dataset.replace('folktables_design_a', 'FT-A').replace('folktables_design_b', 'FT-B').replace('biobias', 'BioBias').replace('atus', 'ATUS')} {'' if pd.isna(row.year) else int(row.year)} {row.score.upper()}".strip(),
                axis=1,
            )
        )
    )
    order = labels["label"].tolist()
    mapping = {(row.dataset, row.year if not pd.isna(row.year) else None, row.score): row.label for row in labels.itertuples(index=False)}
    plot = long.copy()
    plot["label"] = [mapping[(row.dataset, row.year if not pd.isna(row.year) else None, row.score)] for row in plot.itertuples(index=False)]
    metrics = (
        ("paired_absolute_conflict_reduction", "Supported conflict-strength reduction"),
        ("raw_reversal_count", "Raw reversal count"),
        ("ess_fraction", "ESS fraction"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    rng = np.random.default_rng(20260804)
    for axis, (metric, title) in zip(axes, metrics):
        for index, label in enumerate(order):
            values = pd.to_numeric(plot.loc[plot["label"] == label, metric], errors="coerce").dropna().to_numpy()
            jitter = rng.uniform(-0.16, 0.16, len(values))
            axis.scatter(values, index + jitter, s=18, alpha=0.58, color="#2563EB", edgecolors="none")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            axis.errorbar(mean, index, xerr=std, fmt="o", color="#B42318", capsize=3, markersize=5)
        axis.axvline(0.0, color="#667085", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Fixed-model calibration/repair/test split robustness: all resplit outcomes")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def failure_summary() -> pd.DataFrame:
    rows = []
    for failure_path in sorted((output_root() / "splits").glob("**/FAILED.json"), key=str):
        payload = json.loads(failure_path.read_text(encoding="utf-8"))
        rows.append({"path": str(failure_path), **payload})
    columns = [
        "path",
        "status",
        "dataset",
        "year",
        "split_seed",
        "error_type",
        "error_message",
        "traceback",
        "runtime_seconds",
    ]
    return pd.DataFrame(rows, columns=columns)


def runtime_summary() -> pd.DataFrame:
    rows = []
    for done_path in sorted((output_root() / "splits").glob("*/seed_*/DONE.json"), key=str):
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    jobs = pd.DataFrame(rows)
    return (
        jobs.groupby(["dataset", "year"], dropna=False)["runtime_seconds"]
        .agg(["count", "mean", "std", "median", "min", "max", "sum"])
        .reset_index()
    )


def aggregate_outputs() -> dict[str, Any]:
    output_dir = output_root() / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    long = load_long_results()
    if long.empty:
        raise RuntimeError("No completed split results are available to aggregate")
    stats = descriptive_statistics(long)
    frequencies = frequency_summary(long)
    appendix = appendix_ready(long, stats, frequencies)
    failures = failure_summary()
    runtime = runtime_summary()
    long.to_csv(output_dir / "table1_per_split_long.csv", index=False)
    stats.merge(frequencies, on=["dataset", "year", "score"], how="left").to_csv(
        output_dir / "table1_split_summary.csv", index=False
    )
    runtime.to_csv(output_dir / "table1_runtime_summary.csv", index=False)
    failures.to_csv(output_dir / "table1_failure_summary.csv", index=False)
    appendix.to_csv(output_dir / "table1_appendix_ready.csv", index=False)
    write_latex_table(appendix, output_dir / "table1_appendix_ready.tex")
    make_robustness_figure(long, output_dir / "table1_split_robustness.pdf")
    return {
        "n_rows": len(long),
        "n_experiments": len(appendix),
        "n_failures": len(failures),
        "successful_split_seeds": sorted(int(seed) for seed in long["split_seed"].unique()),
    }
