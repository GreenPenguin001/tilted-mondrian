#!/usr/bin/env python3
"""Summarize completed Design B year-level outputs without pooling raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curve_utils import jdump, mkdir, read_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "design_config.yaml"
STRENGTH_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]
SET_SIZE_ALPHAS = [0.01, 0.05, 0.10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--years", default=None, help="Comma-separated years. Defaults to design_config.yaml years.")
    p.add_argument("--split", choices=["repair", "test", "all"], default="test", help="Evaluation split to summarize; default is test.")
    p.add_argument("--summary-dir", type=Path, default=SCRIPT_DIR / "cross_year_summary")
    return p.parse_args()


def resolve_path(path: str | Path, base: Path = SCRIPT_DIR) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def parse_years(text: str | None, cfg: dict[str, Any]) -> list[int]:
    if text:
        return [int(x) for x in str(text).split(",") if x.strip()]
    return [int(x) for x in cfg.get("years", [2014, 2015, 2016, 2017, 2018])]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def split_filter(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if df.empty or split == "all" or "split" not in df:
        return df.copy()
    return df[df["split"].astype(str).eq(str(split))].copy()


def first_row(df: pd.DataFrame, split: str) -> dict[str, Any]:
    df = split_filter(df, split)
    if "policy" in df and not df.empty:
        pooled = df[df["policy"].astype(str).eq("pooled")]
        if not pooled.empty:
            df = pooled
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def number(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def min_group_z_for_split(train_dir: Path, split: str) -> float:
    df = read_csv(train_dir / "cell_counts_group_z.csv")
    df = split_filter(df, split)
    if df.empty or "n" not in df:
        return float("nan")
    return float(pd.to_numeric(df["n"], errors="coerce").min())


def first_row_by_threshold(df: pd.DataFrame, split: str) -> dict[float, dict[str, Any]]:
    df = split_filter(df, split)
    out: dict[float, dict[str, Any]] = {}
    if df.empty or "strength_threshold" not in df:
        return out
    for threshold in STRENGTH_THRESHOLDS:
        sub = df[np.isclose(pd.to_numeric(df["strength_threshold"], errors="coerce"), threshold)]
        out[threshold] = sub.iloc[0].to_dict() if not sub.empty else {}
    return out


def material_counts(year_dir: Path, split: str) -> dict[str, Any]:
    df = first_row_by_threshold(read_csv(year_dir / "adjusted_ipw_curve" / "material_strength_counts.csv"), split)
    out: dict[str, Any] = {}
    for threshold, row in df.items():
        key = f"{threshold:.3f}"
        out[f"raw_material_count_gt_{key}"] = integer(row, "raw_material_strength_count")
        out[f"ipw_material_count_gt_{key}"] = integer(row, "ipw_material_strength_count")
        out[f"raw_material_strength_sum_gt_{key}"] = number(row, "raw_material_strength_sum", 0.0)
        out[f"ipw_material_strength_sum_gt_{key}"] = number(row, "ipw_material_strength_sum", 0.0)
        out[f"material_count_reduction_gt_{key}"] = out[f"raw_material_count_gt_{key}"] - out[f"ipw_material_count_gt_{key}"]
    return out


def persistent_counts(year_dir: Path, split: str) -> dict[str, int]:
    comp = split_filter(read_csv(year_dir / "adjusted_ipw_curve" / "persistent_reversals_comparison.csv"), split)
    if not comp.empty:
        raw = int((pd.to_numeric(comp.get("num_sign_reversal_alpha_raw", 0), errors="coerce").fillna(0) > 0).sum())
        ipw = int((pd.to_numeric(comp.get("num_sign_reversal_alpha_ipw", 0), errors="coerce").fillna(0) > 0).sum())
        return {"raw_persistent_sign_reversal_count": raw, "ipw_persistent_sign_reversal_count": ipw}
    raw_df = split_filter(read_csv(year_dir / "raw_curve" / "persistent_reversals.csv"), split)
    ipw_df = split_filter(read_csv(year_dir / "adjusted_ipw_curve" / "persistent_reversals_ipw.csv"), split)
    raw = int((pd.to_numeric(raw_df.get("num_sign_reversal_alpha", 0), errors="coerce").fillna(0) > 0).sum()) if not raw_df.empty else 0
    ipw = int((pd.to_numeric(ipw_df.get("num_sign_reversal_alpha", 0), errors="coerce").fillna(0) > 0).sum()) if not ipw_df.empty else 0
    return {"raw_persistent_sign_reversal_count": raw, "ipw_persistent_sign_reversal_count": ipw}


def avg_set_sizes(year_dir: Path, split: str) -> dict[str, float]:
    curve = split_filter(read_csv(year_dir / "raw_curve" / "curve_aggregate.csv"), split)
    out: dict[str, float] = {}
    if curve.empty:
        for alpha in SET_SIZE_ALPHAS:
            out[f"avg_set_size_alpha_{alpha:.2f}"] = float("nan")
        return out
    for alpha in SET_SIZE_ALPHAS:
        sub = curve[np.isclose(pd.to_numeric(curve["alpha"], errors="coerce"), alpha)]
        if sub.empty:
            out[f"avg_set_size_alpha_{alpha:.2f}"] = float("nan")
            continue
        n = pd.to_numeric(sub["n"], errors="coerce").fillna(0.0).to_numpy(float)
        v = pd.to_numeric(sub["avg_set_size"], errors="coerce").to_numpy(float)
        denom = float(n.sum())
        out[f"avg_set_size_alpha_{alpha:.2f}"] = float(np.nansum(n * v) / denom) if denom > 0 else float("nan")
    return out


def summarize_year(year: int, out_root: Path, split: str) -> dict[str, Any]:
    year_dir = out_root / str(year)
    train_dir = year_dir / "train"
    manifest = load_json(train_dir / "manifest.json")
    screening = load_json(train_dir / "screening_summary.json")
    raw_args = load_json(year_dir / "raw_curve" / "args.json")
    metrics = load_json(train_dir / f"{split}_metrics.json") if split != "all" else load_json(train_dir / "test_metrics.json")

    rate = first_row(read_csv(year_dir / "adjusted_ipw_curve" / "reversal_rate_summary.csv"), split)
    strength = first_row(read_csv(year_dir / "adjusted_ipw_curve" / "conflict_strength_summary.csv"), split)
    budget = first_row(read_csv(year_dir / "adjusted_ipw_curve" / "alignment_budget_summary.csv"), split)
    stability = first_row(read_csv(year_dir / "adjusted_ipw_curve" / "stability_summary.csv"), split)

    split_sizes = manifest.get("split_sizes") or screening.get("split_sizes") or {}
    n_after_labeling_before_cap = manifest.get("n_after_labeling_before_cap", screening.get("n_after_labeling_before_cap"))
    n_after_group_filter = manifest.get("n_after_group_filter", screening.get("n_after_group_filter"))
    n_before_cap = manifest.get("n_before_cap", screening.get("n_before_cap", n_after_labeling_before_cap))
    n_after_cap = manifest.get("n_after_cap", screening.get("n_after_cap"))
    cap_applied = boolish(manifest.get("cap_applied", screening.get("cap_applied", False)))
    max_final_rows = manifest.get("max_final_rows", screening.get("max_final_rows"))
    states_loaded = manifest.get("states_loaded", screening.get("states_loaded", []))
    thresholds = manifest.get("income_quintile_thresholds", screening.get("income_quintile_thresholds", []))

    row: dict[str, Any] = {
        "year": year,
        "status": "ok" if train_dir.exists() and (year_dir / "raw_curve").exists() and (year_dir / "adjusted_ipw_curve").exists() else "missing",
        "split": split,
        "states_requested": manifest.get("states_requested", screening.get("states_requested")),
        "states_loaded_count": len(states_loaded) if isinstance(states_loaded, list) else float("nan"),
        "used_all_states": manifest.get("states_requested", screening.get("states_requested")) == "all",
        "group_mode": manifest.get("group_mode", screening.get("group_mode")),
        "feature_policy": manifest.get("feature_policy", screening.get("feature_policy")),
        "z_in_x": manifest.get("z_in_x", screening.get("z_in_x")),
        "schl_in_x": manifest.get("schl_in_x", screening.get("schl_in_x")),
        "income_proxy_removed": manifest.get("income_proxy_removed", screening.get("income_proxy_removed")),
        "n_raw_before_filter": manifest.get("n_raw_before_filter", screening.get("n_raw_before_filter")),
        "n_after_valid_filter": manifest.get("n_after_valid_filter", screening.get("n_after_valid_filter")),
        "n_after_group_filter": n_after_group_filter,
        "n_before_cap": n_before_cap,
        "n_after_labeling_before_cap": n_after_labeling_before_cap,
        "max_final_rows": max_final_rows,
        "n_after_cap": n_after_cap,
        "cap_strategy": manifest.get("cap_strategy", screening.get("cap_strategy")),
        "cap_seed": manifest.get("cap_seed", screening.get("cap_seed")),
        "cap_applied": cap_applied,
        "split_train": split_sizes.get("train"),
        "split_cal": split_sizes.get("cal"),
        "split_repair": split_sizes.get("repair"),
        "split_test": split_sizes.get("test"),
        "min_n_group_z_test": min_group_z_for_split(train_dir, "test"),
        "income_quintile_thresholds": thresholds,
        "income_quintile_threshold_scope": manifest.get("income_quintile_threshold_scope", screening.get("income_quintile_threshold_scope")),
        "test_accuracy": metrics.get("accuracy"),
        "test_top3_accuracy": metrics.get("top3_accuracy"),
        "test_log_loss": metrics.get("log_loss"),
        "alpha_grid_count": raw_args.get("alpha_grid_count"),
        "alpha_min": raw_args.get("alpha_min_resolved"),
        "alpha_max": raw_args.get("alpha_max_resolved"),
        "alpha_step": raw_args.get("alpha_step"),
        "raw_sign_reversal_count": integer(rate, "raw_sign_reversal_count"),
        "raw_sign_reversal_rate": number(rate, "raw_sign_reversal_rate"),
        "ipw_sign_reversal_count": integer(rate, "ipw_sign_reversal_count"),
        "ipw_sign_reversal_rate": number(rate, "ipw_sign_reversal_rate"),
        "raw_conflict_strength_sum": number(strength, "raw_conflict_strength_sum", 0.0),
        "raw_conflict_strength_mean": number(strength, "raw_conflict_strength_mean_over_active_alpha_cells", 0.0),
        "ipw_conflict_strength_sum": number(strength, "ipw_conflict_strength_sum", 0.0),
        "ipw_conflict_strength_mean": number(strength, "ipw_conflict_strength_mean_over_active_alpha_cells", 0.0),
        "integrated_composition_reduction": number(budget, "integrated_composition_reduction", 0.0),
        "integrated_variability_price": number(budget, "integrated_variability_price", 0.0),
        "integrated_alignment_budget_gain": number(budget, "integrated_alignment_budget_gain", 0.0),
        "positive_budget_alpha_fraction": number(budget, "positive_budget_alpha_fraction"),
        "tv_z_by_group_before": number(stability, "audit_cell_tv_before"),
        "tv_z_by_group_after": number(stability, "audit_cell_tv_after"),
        "max_profile_smd_before": number(stability, "max_abs_smd_before"),
        "max_profile_smd_after": number(stability, "max_abs_smd_after"),
        "mean_profile_smd_before": number(stability, "mean_abs_smd_before"),
        "mean_profile_smd_after": number(stability, "mean_abs_smd_after"),
        "ess": number(stability, "ess"),
        "ess_fraction": number(stability, "ess_fraction"),
        "weight_p95": number(stability, "p95"),
        "weight_p99": number(stability, "p99"),
        "weight_max": number(stability, "max"),
        "weight_cv": number(stability, "cv"),
        "cap_fraction": number(stability, "cap_fraction"),
    }
    row.update(avg_set_sizes(year_dir, split))
    row.update(persistent_counts(year_dir, split))
    row.update(material_counts(year_dir, split))
    row["sign_rate_reduction"] = row["raw_sign_reversal_rate"] - row["ipw_sign_reversal_rate"]
    row["conflict_strength_reduction"] = row["raw_conflict_strength_sum"] - row["ipw_conflict_strength_sum"]
    row["high_strength_count_reduction_gt_0.020"] = row.get("material_count_reduction_gt_0.020", 0)
    row["tv_reduction"] = row["tv_z_by_group_before"] - row["tv_z_by_group_after"]
    row["supports_composition_driven_mitigation"] = bool(
        row["sign_rate_reduction"] > 0
        and row["conflict_strength_reduction"] > 0
        and row["integrated_composition_reduction"] > 0
        and row["integrated_alignment_budget_gain"] > 0
        and row["tv_reduction"] > 0
    )
    return row


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def year_report_text(row: dict[str, Any]) -> str:
    year = int(row["year"])
    split_sizes = (
        f"train={row.get('split_train')}, cal={row.get('split_cal')}, "
        f"repair={row.get('split_repair')}, test={row.get('split_test')}"
    )
    thresholds = row.get("income_quintile_thresholds", [])
    threshold_text = ", ".join(fmt(float(x), 2) for x in thresholds) if thresholds else "NA"
    lines = [
        f"# Design B {year} Report",
        "",
        "## Scope",
        "",
        f"- Year/split: {year}, `{row.get('split')}`",
        f"- All states: {row.get('used_all_states')} (`states_loaded_count={row.get('states_loaded_count')}`)",
        f"- group_mode: `{row.get('group_mode')}`",
        f"- feature_policy: `{row.get('feature_policy')}`",
        f"- z_in_x: {row.get('z_in_x')} (`SCHL in X={row.get('schl_in_x')}`)",
        f"- income proxy removed from X: {row.get('income_proxy_removed')}",
        f"- max_final_rows cap applied: {row.get('cap_applied')} (`max_final_rows={row.get('max_final_rows')}`)",
        f"- cap strategy/seed: `{row.get('cap_strategy')}` / `{row.get('cap_seed')}`",
        "- Task: Y=PINCP income quintile, G=RAC1P White vs Black, Z=SCHL education bucket.",
        "- Propensity: logistic group propensity overlap IPW, e(H)=P(G=1|H), fit on repair split.",
        "- Boundary: H uses only frozen softmax prediction-profile features; Y/Z/set size/alpha outcomes/threshold artifacts are not propensity inputs.",
        "",
        "## Data",
        "",
        f"- n_raw_before_filter: {row.get('n_raw_before_filter')}",
        f"- n_after_valid_filter: {row.get('n_after_valid_filter')}",
        f"- n_after_group_filter: {row.get('n_after_group_filter')}",
        f"- n_before_cap: {row.get('n_before_cap')}",
        f"- n_after_cap: {row.get('n_after_cap')}",
        f"- split sizes: {split_sizes}",
        f"- min n(G,Z) on test: {fmt(row.get('min_n_group_z_test'), 0)}",
        f"- income quintile thresholds: {threshold_text}",
        f"- income threshold scope: `{row.get('income_quintile_threshold_scope')}`",
        "",
        "## Prediction",
        "",
        f"- test accuracy: {fmt(row.get('test_accuracy'))}",
        f"- test top3 accuracy: {fmt(row.get('test_top3_accuracy'))}",
        f"- test log loss: {fmt(row.get('test_log_loss'))}",
        f"- average set size at alpha=0.01/0.05/0.10: {fmt(row.get('avg_set_size_alpha_0.01'))} / {fmt(row.get('avg_set_size_alpha_0.05'))} / {fmt(row.get('avg_set_size_alpha_0.10'))}",
        "",
        "## Alpha Grid",
        "",
        f"- points: {row.get('alpha_grid_count')}",
        f"- min/max/step: {fmt(row.get('alpha_min'), 3)} / {fmt(row.get('alpha_max'), 3)} / {fmt(row.get('alpha_step'), 3)}",
        "",
        "## Count Diagnostics",
        "",
        "| metric | raw | IPW |",
        "|---|---:|---:|",
        f"| sign reversal count | {row.get('raw_sign_reversal_count')} | {row.get('ipw_sign_reversal_count')} |",
        f"| sign reversal rate | {fmt(row.get('raw_sign_reversal_rate'))} | {fmt(row.get('ipw_sign_reversal_rate'))} |",
        f"| persistent sign reversal count | {row.get('raw_persistent_sign_reversal_count')} | {row.get('ipw_persistent_sign_reversal_count')} |",
        "",
        "## Strength-Level Diagnostics",
        "",
        "| metric | raw | IPW |",
        "|---|---:|---:|",
        f"| conflict_strength_sum | {fmt(row.get('raw_conflict_strength_sum'))} | {fmt(row.get('ipw_conflict_strength_sum'))} |",
        f"| conflict_strength_mean | {fmt(row.get('raw_conflict_strength_mean'))} | {fmt(row.get('ipw_conflict_strength_mean'))} |",
        "",
        "| threshold | raw count | IPW count | reduction |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in STRENGTH_THRESHOLDS:
        key = f"{threshold:.3f}"
        lines.append(
            f"| {key} | {row.get(f'raw_material_count_gt_{key}', 'NA')} | "
            f"{row.get(f'ipw_material_count_gt_{key}', 'NA')} | "
            f"{row.get(f'material_count_reduction_gt_{key}', 'NA')} |"
        )
    lines.extend([
        "",
        "## Alignment Budget",
        "",
        f"- integrated composition_reduction: {fmt(row.get('integrated_composition_reduction'))}",
        f"- integrated variability_price: {fmt(row.get('integrated_variability_price'))}",
        f"- integrated alignment_budget_gain: {fmt(row.get('integrated_alignment_budget_gain'))}",
        f"- positive_budget_alpha_fraction: {fmt(row.get('positive_budget_alpha_fraction'))}",
        "",
        "## Composition And Stability",
        "",
        f"- TV(P(Z|G)) before/after: {fmt(row.get('tv_z_by_group_before'))} / {fmt(row.get('tv_z_by_group_after'))}",
        f"- max profile SMD before/after: {fmt(row.get('max_profile_smd_before'))} / {fmt(row.get('max_profile_smd_after'))}",
        f"- mean profile SMD before/after: {fmt(row.get('mean_profile_smd_before'))} / {fmt(row.get('mean_profile_smd_after'))}",
        f"- ESS / ESS fraction: {fmt(row.get('ess'))} / {fmt(row.get('ess_fraction'))}",
        f"- weight p95/p99/max: {fmt(row.get('weight_p95'))} / {fmt(row.get('weight_p99'))} / {fmt(row.get('weight_max'))}",
        f"- weight CV / cap fraction: {fmt(row.get('weight_cv'))} / {fmt(row.get('cap_fraction'))}",
        "",
        "## Conclusion",
        "",
        f"- Supports composition-driven mitigation: {row.get('supports_composition_driven_mitigation')}",
        "- Main-text wording: IPW mitigates composition-driven Simpson disagreement when composition-residual reduction dominates local-gap variability price; it does not claim universal removal.",
        "",
    ])
    return "\n".join(lines)


def write_cross_year_report(summary_dir: Path, rows: list[dict[str, Any]], medians: dict[str, float], pass_counts: dict[str, int]) -> None:
    lines = [
        "# Design B Cross-Year Summary",
        "",
        "This summary reads completed year-level CSV/JSON outputs and reports the test split by default. It does not pool raw ACS rows.",
        "",
        "| year | raw sign rate | IPW sign rate | conflict reduction | high-strength count reduction >0.020 | composition reduction | variability price | budget gain | TV before | TV after | ESS fraction | cap applied | supports mitigation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['year']} | {fmt(row.get('raw_sign_reversal_rate'))} | {fmt(row.get('ipw_sign_reversal_rate'))} | "
            f"{fmt(row.get('conflict_strength_reduction'))} | {row.get('high_strength_count_reduction_gt_0.020')} | "
            f"{fmt(row.get('integrated_composition_reduction'))} | {fmt(row.get('integrated_variability_price'))} | "
            f"{fmt(row.get('integrated_alignment_budget_gain'))} | {fmt(row.get('tv_z_by_group_before'))} | "
            f"{fmt(row.get('tv_z_by_group_after'))} | {fmt(row.get('ess_fraction'))} | {row.get('cap_applied')} | "
            f"{row.get('supports_composition_driven_mitigation')} |"
        )
    lines.extend(["", "## Medians", ""])
    for key, value in medians.items():
        lines.append(f"- {key}: {fmt(value)}")
    lines.extend(["", "## Year Pass Counts", ""])
    for key, value in pass_counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    (summary_dir / "design_B_cross_year_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    years = parse_years(args.years, cfg)
    out_root = resolve_path(args.output_root)
    summary_dir = mkdir(resolve_path(args.summary_dir))

    rows = [summarize_year(year, out_root, args.split) for year in years]
    for row in rows:
        if row.get("status") == "ok":
            (out_root / f"{int(row['year'])}_report.md").write_text(year_report_text(row), encoding="utf-8")

    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / "design_B_cross_year_summary.csv", index=False)
    df.to_csv(summary_dir / "year_summary.csv", index=False)

    ok = df[df["status"].eq("ok")].copy() if "status" in df else pd.DataFrame()
    median_cols = [
        "raw_sign_reversal_rate",
        "ipw_sign_reversal_rate",
        "conflict_strength_reduction",
        "high_strength_count_reduction_gt_0.020",
        "integrated_composition_reduction",
        "integrated_variability_price",
        "integrated_alignment_budget_gain",
        "positive_budget_alpha_fraction",
        "tv_z_by_group_before",
        "tv_z_by_group_after",
        "ess_fraction",
    ]
    medians = {
        col: float(pd.to_numeric(ok[col], errors="coerce").median())
        for col in median_cols
        if not ok.empty and col in ok
    }
    pass_counts = {
        "years_summarized": int(len(ok)),
        "sign_rate_reduction_positive": int((pd.to_numeric(ok.get("sign_rate_reduction", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "conflict_strength_reduction_positive": int((pd.to_numeric(ok.get("conflict_strength_reduction", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "high_strength_count_reduction_gt_0.020_positive": int((pd.to_numeric(ok.get("high_strength_count_reduction_gt_0.020", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "integrated_composition_reduction_positive": int((pd.to_numeric(ok.get("integrated_composition_reduction", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "integrated_alignment_budget_gain_positive": int((pd.to_numeric(ok.get("integrated_alignment_budget_gain", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "tv_reduction_positive": int((pd.to_numeric(ok.get("tv_reduction", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not ok.empty else 0,
        "ess_fraction_ge_0.50": int((pd.to_numeric(ok.get("ess_fraction", pd.Series(dtype=float)), errors="coerce") >= 0.50).sum()) if not ok.empty else 0,
        "supports_composition_driven_mitigation": int(ok.get("supports_composition_driven_mitigation", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not ok.empty else 0,
        "row_cap_applied": int(ok.get("cap_applied", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not ok.empty else 0,
    }
    summary = {
        "design_name": cfg.get("design_name", "design_B_income_education"),
        "years_requested": years,
        "summary_split": args.split,
        "years_summarized": [int(x) for x in ok["year"].to_numpy()] if not ok.empty else [],
        "years_missing": [int(r["year"]) for r in rows if r.get("status") != "ok"],
        "median_metrics": medians,
        "year_pass_counts": pass_counts,
        "records": rows,
        "note": "Cross-year summary reads completed year-level CSV/JSON outputs only and does not pool ACS rows or prediction arrays.",
    }
    jdump(summary, summary_dir / "design_B_cross_year_summary.json")
    jdump(summary, summary_dir / "summary.json")
    write_cross_year_report(summary_dir, rows, medians, pass_counts)
    print(f"Wrote cross-year summary to {summary_dir}", flush=True)


if __name__ == "__main__":
    main()
