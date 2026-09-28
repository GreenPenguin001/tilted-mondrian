#!/usr/bin/env python3
"""Stage 2: compute raw split-conformal set-size curves and Simpson diagnostics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from curve_utils import (
    aggregate_gap,
    alignment_budget,
    alignment_budget_summary,
    alignment_components,
    alignment_summary,
    alpha_block_summary,
    alpha_grid_from_range,
    audit_name,
    build_thresholds,
    covered,
    group_curve,
    conflict_strength_by_alpha,
    conflict_strength_summary,
    jdump,
    jload,
    local_curve_by_z,
    local_gap,
    mkdir,
    parse_alpha_grid,
    persistent_reversals,
    read_yaml,
    reversal_rate_summary,
    reversal_counts_by_alpha,
    scores_from_probs,
    set_sizes,
    simpson_reversal_table,
    material_strength_counts,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "design_config.yaml"


def resolve_path(path: str | Path, base: Path = SCRIPT_DIR) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--split", choices=["repair", "test", "all"], default="test")
    p.add_argument("--alpha-grid", default=None)
    p.add_argument("--alpha_min", type=float, default=0.02)
    p.add_argument("--alpha_max", type=float, default=0.10)
    p.add_argument("--alpha_step", type=float, default=0.005)
    p.add_argument("--alpha_grid_name", default=None)
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--alpha0", type=float, default=None)
    return p.parse_args()


def load_split(train_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "probs": np.load(train_dir / f"probs_{split}.npy"),
        "y": np.load(train_dir / f"y_{split}.npy"),
        "g": np.load(train_dir / f"g_{split}.npy"),
        "z": np.load(train_dir / f"z_{split}.npy"),
    }


def split_list(requested: str) -> list[str]:
    return ["repair", "test"] if requested == "all" else [requested]


def alpha_grid_for_args(args: argparse.Namespace, conformal_cfg: dict[str, Any]) -> list[float]:
    if args.alpha_grid is not None:
        return parse_alpha_grid(args.alpha_grid)
    if (
        args.alpha_min == 0.02
        and args.alpha_max == 0.10
        and args.alpha_step == 0.005
        and "alpha_grid" in conformal_cfg
    ):
        return parse_alpha_grid(conformal_cfg.get("alpha_grid", "dense_safe"))
    return alpha_grid_from_range(args.alpha_min, args.alpha_max, args.alpha_step)


def output_dir_for_grid(year_dir: Path, base_name: str, alpha_grid_name: str | None) -> Path:
    if alpha_grid_name is None or str(alpha_grid_name).strip() == "":
        return mkdir(year_dir / base_name)
    suffix = str(alpha_grid_name).strip()
    if "/" in suffix or "\\" in suffix:
        raise ValueError("--alpha_grid_name may not contain path separators")
    return mkdir(year_dir / f"{base_name}_{suffix}")


def difficulty_by_z(
    split_name: str,
    probs: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    thresholds: dict[float, float],
    alphas: list[float],
    label_maps: dict[str, Any],
    score: str,
) -> pd.DataFrame:
    scores = scores_from_probs(probs, score)
    z = np.asarray(z, dtype=int)
    y = np.asarray(y, dtype=int)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        threshold = float(thresholds[float(alpha)])
        size = set_sizes(scores, threshold)
        cov = covered(scores, y, threshold)
        for audit in sorted(int(x) for x in np.unique(z)):
            mask = z == audit
            rows.append({
                "split": split_name,
                "policy": "pooled",
                "alpha": float(alpha),
                "audit": int(audit),
                "audit_name": audit_name(label_maps, audit),
                "n": int(mask.sum()),
                "mean_set_size": float(np.mean(size[mask])) if mask.any() else float("nan"),
                "coverage": float(np.mean(cov[mask])) if mask.any() else float("nan"),
                "threshold": threshold,
            })
    return pd.DataFrame(rows)


def difficulty_variation_summary(difficulty: pd.DataFrame, alpha0: float) -> dict[str, Any]:
    if difficulty.empty:
        return {"rows": 0, "by_alpha": [], "by_split_alpha0": []}
    by_alpha: list[dict[str, Any]] = []
    by_split_alpha0: list[dict[str, Any]] = []
    for (split, policy, alpha), sub in difficulty.groupby(["split", "policy", "alpha"], sort=True):
        vals = pd.to_numeric(sub["mean_set_size"], errors="coerce")
        row = {
            "split": str(split),
            "policy": str(policy),
            "alpha": float(alpha),
            "num_audit_cells": int(vals.notna().sum()),
            "mean_across_z": float(vals.mean()),
            "std_across_z": float(vals.std(ddof=0)),
            "min_across_z": float(vals.min()),
            "max_across_z": float(vals.max()),
            "range_across_z": float(vals.max() - vals.min()),
        }
        if vals.notna().any():
            row["min_audit"] = str(sub.loc[vals.idxmin(), "audit_name"])
            row["max_audit"] = str(sub.loc[vals.idxmax(), "audit_name"])
        by_alpha.append(row)
    for (split, policy), sub in difficulty.groupby(["split", "policy"], sort=True):
        nearest_alpha = float(sub.iloc[int(np.argmin(np.abs(sub["alpha"].to_numpy(float) - float(alpha0))))]["alpha"])
        near = sub[np.isclose(sub["alpha"].to_numpy(float), nearest_alpha)]
        vals = pd.to_numeric(near["mean_set_size"], errors="coerce")
        by_split_alpha0.append({
            "split": str(split),
            "policy": str(policy),
            "alpha0": float(alpha0),
            "nearest_alpha": nearest_alpha,
            "num_audit_cells": int(vals.notna().sum()),
            "mean_across_z": float(vals.mean()),
            "std_across_z": float(vals.std(ddof=0)),
            "min_across_z": float(vals.min()),
            "max_across_z": float(vals.max()),
            "range_across_z": float(vals.max() - vals.min()),
        })
    return {"rows": int(len(difficulty)), "by_alpha": by_alpha, "by_split_alpha0": by_split_alpha0}


def summarize_curve(reversals: pd.DataFrame, counts: pd.DataFrame, persistent: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    if not reversals.empty:
        for split, sub in reversals.groupby("split"):
            supported = sub[sub["supported"]]
            sign_active = supported[supported["is_sign_reversal"]]
            supported_cells = int(supported.shape[0])
            conflict_sum = float(pd.to_numeric(supported["conflict_strength"], errors="coerce").sum()) if supported_cells else 0.0
            by_split[str(split)] = {
                "supported_alpha_cells": supported_cells,
                "sign_reversal_count": int(sign_active.shape[0]),
                "sign_reversal_rate": float(sign_active.shape[0] / supported_cells) if supported_cells else float("nan"),
                "persistent_sign_reversal_count": int((persistent[persistent["split"] == split]["num_sign_reversal_alpha"] > 0).sum()) if not persistent.empty else 0,
                "conflict_strength_sum": conflict_sum,
                "conflict_strength_mean_over_active_alpha_cells": float(conflict_sum / supported_cells) if supported_cells else float("nan"),
                "persistent_sign_reversals": int((persistent[persistent["split"] == split]["num_sign_reversal_alpha"] > 0).sum()) if not persistent.empty else 0,
                "persistent_audit_cells": int((persistent[persistent["split"] == split]["num_sign_reversal_alpha"] > 0).sum()) if not persistent.empty else 0,
            }
    return {
        "stage": "raw_curve",
        "year": int(args.year),
        "requested_split": args.split,
        "main_simpson_count": "sign_reversal_count",
        "diagnostic_classes": [
            "count",
            "strength-level count",
            "composition effect",
            "variance/weight stability",
        ],
        "by_split": by_split,
        "counts_rows": int(len(counts)),
        "persistent_rows": int(len(persistent)),
        "validity_boundary": "Raw curves summarize unweighted set sizes from ordinary split-conformal thresholds.",
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    start = time.time()
    out_root = resolve_path(args.output_root)
    train_dir = out_root / str(args.year) / "train"
    out_dir = output_dir_for_grid(out_root / str(args.year), "raw_curve", args.alpha_grid_name)
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    label_maps = jload(train_dir / "label_maps.json")
    cal = load_split(train_dir, "cal")
    conformal_cfg = cfg.get("conformal", {})
    diagnostics_cfg = cfg.get("diagnostics", {})
    score = str(conformal_cfg.get("score", "simple"))
    alphas = alpha_grid_for_args(args, conformal_cfg)
    n_min = int(args.n_min if args.n_min is not None else diagnostics_cfg.get("n_min", 200))
    alpha0 = float(args.alpha0 if args.alpha0 is not None else diagnostics_cfg.get("alpha0", 0.10))

    threshold_df, thresholds = build_thresholds(cal["probs"], cal["y"], alphas, score)
    threshold_df.to_csv(out_dir / "conformal_thresholds.csv", index=False)

    aggregate_tables: list[pd.DataFrame] = []
    local_tables: list[pd.DataFrame] = []
    difficulty_tables: list[pd.DataFrame] = []
    for split in split_list(args.split):
        data = load_split(train_dir, split)
        weights = np.ones(len(data["y"]), dtype=float)
        aggregate_tables.append(
            group_curve(
                split, data["probs"], data["y"], data["g"], data["z"],
                thresholds, alphas, label_maps, weights=weights, score=score, estimand="raw",
            )
        )
        local_tables.append(
            local_curve_by_z(
                split, data["probs"], data["y"], data["g"], data["z"],
                thresholds, alphas, label_maps, weights=weights, score=score, estimand="raw",
            )
        )
        difficulty_tables.append(
            difficulty_by_z(split, data["probs"], data["y"], data["z"], thresholds, alphas, label_maps, score)
        )

    aggregate = pd.concat(aggregate_tables, ignore_index=True)
    local = pd.concat(local_tables, ignore_index=True)
    difficulty = pd.concat(difficulty_tables, ignore_index=True)
    gap = aggregate_gap(aggregate)
    gap_z = local_gap(local)
    reversals = simpson_reversal_table(gap, gap_z, n_min=n_min)
    counts = reversal_counts_by_alpha(reversals)
    persistent = persistent_reversals(reversals, alpha0=alpha0)
    align = alignment_summary(reversals, "raw")
    rate_summary = reversal_rate_summary({"raw": reversals})
    strength_summary = conflict_strength_summary({"raw": reversals})
    strength_by_alpha = conflict_strength_by_alpha({"raw": reversals})
    material_summary = material_strength_counts({"raw": reversals})
    block_summary = alpha_block_summary({"raw": reversals})
    raw_components = alignment_components(aggregate, local, "raw")
    budget = alignment_budget(raw_components)
    budget_summary = alignment_budget_summary(budget)

    aggregate.to_csv(out_dir / "curve_aggregate.csv", index=False)
    local.to_csv(out_dir / "curve_local_by_z.csv", index=False)
    difficulty.to_csv(out_dir / "difficulty_by_z.csv", index=False)
    gap.to_csv(out_dir / "gap_aggregate.csv", index=False)
    gap_z.to_csv(out_dir / "gap_local_by_z.csv", index=False)
    reversals.to_csv(out_dir / "simpson_reversals.csv", index=False)
    counts.to_csv(out_dir / "reversal_counts_by_alpha.csv", index=False)
    persistent.to_csv(out_dir / "persistent_reversals.csv", index=False)
    align.to_csv(out_dir / "alignment_summary.csv", index=False)
    rate_summary.to_csv(out_dir / "reversal_rate_summary.csv", index=False)
    strength_summary.to_csv(out_dir / "conflict_strength_summary.csv", index=False)
    strength_by_alpha.to_csv(out_dir / "conflict_strength_by_alpha.csv", index=False)
    material_summary.to_csv(out_dir / "material_strength_counts.csv", index=False)
    block_summary.to_csv(out_dir / "alpha_block_summary.csv", index=False)
    budget.to_csv(out_dir / "alignment_budget_by_alpha.csv", index=False)
    budget_summary.to_csv(out_dir / "alignment_budget_summary.csv", index=False)

    args_record = vars(args).copy()
    args_record["config"] = str(args.config)
    args_record["output_root"] = str(out_root)
    args_record["alpha_grid_resolved"] = alphas
    args_record["alpha_grid_count"] = len(alphas)
    args_record["alpha_min_resolved"] = min(alphas)
    args_record["alpha_max_resolved"] = max(alphas)
    args_record["n_min_resolved"] = n_min
    args_record["weighting_convention"] = "symmetric_group_cell_law_with_pooled_group_prevalence"
    jdump(args_record, out_dir / "args.json")

    summary = summarize_curve(reversals, counts, persistent, args)
    summary["elapsed_seconds"] = time.time() - start
    summary["alpha_grid"] = alphas
    summary["alpha_grid_count"] = len(alphas)
    summary["alpha_min_resolved"] = min(alphas)
    summary["alpha_max_resolved"] = max(alphas)
    summary["alpha_grid_name"] = args.alpha_grid_name
    summary["difficulty_variation"] = difficulty_variation_summary(difficulty, alpha0)
    summary["dense_grid_diagnostics"] = {
        "reversal_rate_summary": rate_summary.to_dict(orient="records"),
        "conflict_strength_summary": strength_summary.to_dict(orient="records"),
        "material_strength_counts": material_summary.to_dict(orient="records"),
        "alignment_summary": align.to_dict(orient="records"),
        "alpha_block_summary": block_summary.to_dict(orient="records"),
    }
    summary["alignment_budget_summary"] = budget_summary.to_dict(orient="records")
    jdump(summary, out_dir / "curve_summary.json")
    jdump(difficulty_variation_summary(difficulty, alpha0), out_dir / "difficulty_variation_summary.json")
    jdump({"rows": align.to_dict(orient="records")}, out_dir / "alignment_summary.json")
    jdump({"rows": rate_summary.to_dict(orient="records")}, out_dir / "reversal_rate_summary.json")
    jdump({"rows": strength_summary.to_dict(orient="records")}, out_dir / "conflict_strength_summary.json")
    jdump({"rows": budget_summary.to_dict(orient="records")}, out_dir / "alignment_budget_summary.json")
    print(f"Wrote raw curve outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
