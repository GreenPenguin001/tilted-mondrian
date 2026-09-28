#!/usr/bin/env python3
"""Run all Design A stages for one ACS year."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "design_config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--split", choices=["repair", "test", "all"], default="test")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_final_rows", type=int, default=None)
    p.add_argument("--cap_seed", type=int, default=None)
    p.add_argument("--cap_strategy", default=None, choices=["stratified_y_g_z"])
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--z-in-x", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--alpha-grid", default=None)
    p.add_argument("--alpha_min", type=float, default=None)
    p.add_argument("--alpha_max", type=float, default=None)
    p.add_argument("--alpha_step", type=float, default=None)
    p.add_argument("--alpha_grid_name", default=None)
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--alpha0", type=float, default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-raw", action="store_true")
    p.add_argument("--skip-ipw", action="store_true")
    return p.parse_args()


def add_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_optional_bool(cmd: list[str], true_flag: str, false_flag: str, value: bool | None) -> None:
    if value is True:
        cmd.append(true_flag)
    elif value is False:
        cmd.append(false_flag)


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main() -> None:
    args = parse_args()
    common = ["--year", str(args.year), "--config", str(args.config), "--output-root", str(args.output_root)]

    if not args.skip_train:
        cmd = [sys.executable, str(SCRIPT_DIR / "01_train_export.py"), *common]
        add_optional(cmd, "--max_rows", args.max_rows)
        add_optional(cmd, "--data-root", args.data_root)
        add_optional(cmd, "--max_final_rows", args.max_final_rows)
        add_optional(cmd, "--cap_seed", args.cap_seed)
        add_optional(cmd, "--cap_strategy", args.cap_strategy)
        add_optional(cmd, "--top-k", args.top_k)
        add_optional(cmd, "--density", args.density)
        add_optional_bool(cmd, "--z-in-x", "--no-z-in-x", args.z_in_x)
        add_optional_bool(cmd, "--download", "--no-download", args.download)
        run(cmd)

    if not args.skip_raw:
        cmd = [sys.executable, str(SCRIPT_DIR / "02_raw_curve.py"), *common, "--split", args.split]
        add_optional(cmd, "--alpha-grid", args.alpha_grid)
        add_optional(cmd, "--alpha_min", args.alpha_min)
        add_optional(cmd, "--alpha_max", args.alpha_max)
        add_optional(cmd, "--alpha_step", args.alpha_step)
        add_optional(cmd, "--alpha_grid_name", args.alpha_grid_name)
        add_optional(cmd, "--n-min", args.n_min)
        add_optional(cmd, "--alpha0", args.alpha0)
        run(cmd)

    if not args.skip_ipw:
        cmd = [sys.executable, str(SCRIPT_DIR / "03_adjusted_ipw_curve.py"), *common, "--split", args.split]
        add_optional(cmd, "--alpha-grid", args.alpha_grid)
        add_optional(cmd, "--alpha_min", args.alpha_min)
        add_optional(cmd, "--alpha_max", args.alpha_max)
        add_optional(cmd, "--alpha_step", args.alpha_step)
        add_optional(cmd, "--alpha_grid_name", args.alpha_grid_name)
        add_optional(cmd, "--n-min", args.n_min)
        add_optional(cmd, "--alpha0", args.alpha0)
        run(cmd)


if __name__ == "__main__":
    main()
