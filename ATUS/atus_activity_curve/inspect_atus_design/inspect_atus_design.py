#!/usr/bin/env python3
"""Inspection-only ATUS design scout.

This script downloads/discovers BLS ATUS 2003-2024 multi-year files and writes
support, label, TV, entropy, and leakage inspection artifacts. It intentionally
does not train models, run conformal prediction, or run IPW.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

BLS_ATUS_0324_FILES = {
    "respondent": "https://www.bls.gov/tus/datafiles/atusresp-0324.zip",
    "activity": "https://www.bls.gov/tus/datafiles/atusact-0324.zip",
    "atus_cps": "https://www.bls.gov/tus/datafiles/atuscps-0324.zip",
    "roster": "https://www.bls.gov/tus/datafiles/atusrost-0324.zip",
    "who": "https://www.bls.gov/tus/datafiles/atuswho-0324.zip",
    "activity_summary": "https://www.bls.gov/tus/datafiles/atussum-0324.zip",
}

REQUIRED_ROLES = {"respondent", "activity"}
RECOMMENDED_ROLES = {"atus_cps", "roster"}
OPTIONAL_ROLES = {"who", "activity_summary"}
TOPK_GRID = [10, 12, 15, 20]
SPLIT_ORDER = ["train", "calibration", "repair", "test"]
SPLIT_FRACTIONS = {
    "train": 0.40,
    "calibration": 0.25,
    "repair": 0.25,
    "test": 0.10,
}

INSPECTION_DESCRIPTION = """Inspection-only ATUS design analysis.
The pipeline uses the BLS multi-year ATUS files and reports:
Use BLS 2003-2024 multi-year ATUS respondent/activity as required files and
atus_cps/roster as strongly recommended context files. Include official
download instructions, an automatic downloader, manual fallback instructions,
unzip, file discovery, reading/merge checks, variable identification, label
support, group support, Z support, TV(P(Z|G)), entropy, candidate summary, and
recommended design JSON. It does not train models or run conformal prediction.

Download command:
python atus_activity_curve/inspect_atus_design/inspect_atus_design.py \\
  --download \\
  --data-dir atus_activity_curve/data/raw_0324 \\
  --extract-dir atus_activity_curve/data/extracted_0324 \\
  --output-dir atus_activity_curve/inspect_atus_design \\
  --seed 4

Manual fallback command:
python atus_activity_curve/inspect_atus_design/inspect_atus_design.py \\
  --skip-download \\
  --data-dir atus_activity_curve/data/raw_0324 \\
  --extract-dir atus_activity_curve/data/extracted_0324 \\
  --output-dir atus_activity_curve/inspect_atus_design \\
  --seed 4
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download BLS ATUS 2003-2024 zip files.")
    parser.add_argument("--skip-download", action="store_true", help="Skip network download and use local files.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "raw_0324")
    parser.add_argument("--extract-dir", type=Path, default=PROJECT_DIR / "data" / "extracted_0324")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def norm_col(name: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).strip().upper())


def find_col(columns: list[str] | pd.Index, candidates: list[str], contains_any: list[str] | None = None) -> str | None:
    """Find a column by exact normalized name, then optional substring tokens."""
    cols = list(columns)
    lookup: dict[str, str] = {}
    for col in cols:
        lookup.setdefault(norm_col(col), col)
    for cand in candidates:
        hit = lookup.get(norm_col(cand))
        if hit is not None:
            return hit
    if contains_any:
        tokens = [norm_col(x) for x in contains_any if str(x).strip()]
        for col in cols:
            ncol = norm_col(col)
            if any(tok and tok in ncol for tok in tokens):
                return col
    return None


def find_cols(columns: list[str] | pd.Index, candidates: list[str], contains_any: list[str] | None = None) -> list[str]:
    cols = list(columns)
    out: list[str] = []
    for cand in candidates:
        hit = find_col(cols, [cand])
        if hit is not None and hit not in out:
            out.append(hit)
    if contains_any:
        tokens = [norm_col(x) for x in contains_any if str(x).strip()]
        for col in cols:
            ncol = norm_col(col)
            if any(tok and tok in ncol for tok in tokens) and col not in out:
                out.append(col)
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None
    return value


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        df = df[columns]
    df.to_csv(path, index=False)
    return df


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def basename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def write_manual_download_instructions(output_dir: Path) -> None:
    text = """# Manual ATUS Download Instructions

Official data page: BLS ATUS 2003-2024 Multi-Year Microdata Files

Official page URL: https://www.bls.gov/tus/datafiles.htm

Download these required/recommended zip files:

- atusresp-0324.zip
- atusact-0324.zip
- atuscps-0324.zip
- atusrost-0324.zip
- optional: atuswho-0324.zip
- optional: atussum-0324.zip

After manual download, place the zip files here:

`atus_activity_curve/data/raw_0324/`

Then run:

```bash
python atus_activity_curve/inspect_atus_design/inspect_atus_design.py \\
  --skip-download \\
  --data-dir atus_activity_curve/data/raw_0324 \\
  --extract-dir atus_activity_curve/data/extracted_0324 \\
  --output-dir atus_activity_curve/inspect_atus_design \\
  --seed 4
```

If the zip files were already extracted elsewhere, place the extracted `.csv`,
`.dat`, or `.txt` microdata files under either `--data-dir` or `--extract-dir`;
the inspection script searches both locations recursively.
"""
    (output_dir / "manual_download_instructions.md").write_text(text, encoding="utf-8")


def write_data_sources(output_dir: Path) -> None:
    rows = ["# ATUS Data Sources", ""]
    rows.append("Official page: https://www.bls.gov/tus/datafiles.htm")
    rows.append("")
    rows.append("BLS 2003-2024 multi-year ATUS microdata files used by the downloader:")
    rows.append("")
    for role, url in BLS_ATUS_0324_FILES.items():
        if role in REQUIRED_ROLES:
            status = "required for main inspection"
        elif role in RECOMMENDED_ROLES:
            status = "strongly recommended context"
        else:
            status = "optional; downloaded/recorded but not primary X"
        rows.append(f"- `{role}`: `{url}` ({status})")
    rows.append("")
    rows.append("Primary files for this design are respondent and activity. CPS and roster are used only to improve respondent/household context discovery. Who-with, where/location, duration, stop time, and activity summary variables are treated as leakage or proxy risks for the primary activity-category design.")
    (output_dir / "data_sources.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_cmd_record(output_dir: Path, argv: list[str]) -> None:
    cmd = " ".join([shlex_quote(sys.executable), *[shlex_quote(x) for x in argv]])
    text = INSPECTION_DESCRIPTION + "\nExecuted command:\n\n```bash\n" + cmd + "\n```\n"
    (output_dir / "cmd_inspect_atus.txt").write_text(text, encoding="utf-8")


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(str(text))


def download_one(url: str, dest: Path, timeout: int = 90) -> tuple[bool, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ATUS-reproduction-script/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with tmp.open("wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
        tmp.replace(dest)
        return True, f"downloaded {dest.name} ({dest.stat().st_size} bytes)"
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def download_files(data_dir: Path, output_dir: Path, should_download: bool, skip_download: bool) -> tuple[pd.DataFrame, list[str], bool]:
    records: list[dict[str, Any]] = []
    missing_after_attempt: list[str] = []
    attempted = should_download and not skip_download
    for role, url in BLS_ATUS_0324_FILES.items():
        filename = basename_from_url(url)
        dest = data_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            records.append({
                "timestamp": now_stamp(),
                "role": role,
                "url": url,
                "path": str(dest),
                "status": "already_present",
                "message": f"Using existing {filename}",
                "bytes": dest.stat().st_size,
            })
            continue
        if skip_download or not should_download:
            status = "skipped"
            message = "Download skipped; local file not found"
            ok = False
        else:
            ok, message = download_one(url, dest)
            status = "downloaded" if ok else "failed"
        records.append({
            "timestamp": now_stamp(),
            "role": role,
            "url": url,
            "path": str(dest),
            "status": status,
            "message": message,
            "bytes": dest.stat().st_size if dest.exists() else 0,
        })
        if role in (REQUIRED_ROLES | RECOMMENDED_ROLES) and (not dest.exists() or dest.stat().st_size == 0):
            missing_after_attempt.append(filename)
    log_df = write_csv(output_dir / "download_log.txt", records)
    automatic_success = attempted and not any(r["status"] == "failed" for r in records) and all((data_dir / basename_from_url(u)).exists() for u in BLS_ATUS_0324_FILES.values())
    return log_df, missing_after_attempt, automatic_success


def safe_extract_zip(zip_path: Path, target_dir: Path) -> tuple[bool, str, int]:
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                dest = (target_dir / member.filename).resolve()
                if not str(dest).startswith(str(target_dir.resolve())):
                    return False, f"unsafe zip member path: {member.filename}", extracted
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                extracted += 1
        return True, f"extracted {extracted} files", extracted
    except zipfile.BadZipFile as exc:
        return False, f"BadZipFile: {exc}", extracted
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", extracted


def extract_archives(data_dir: Path, extract_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for zip_path in sorted(data_dir.rglob("*.zip")):
        target = extract_dir / zip_path.stem
        target.mkdir(parents=True, exist_ok=True)
        ok, message, n_files = safe_extract_zip(zip_path, target)
        records.append({
            "zip_path": str(zip_path),
            "extract_target": str(target),
            "status": "ok" if ok else "failed",
            "n_files": n_files,
            "message": message,
        })
    return pd.DataFrame(records)


def sample_text(path: Path, max_bytes: int = 65536) -> str:
    data = path.read_bytes()[:max_bytes]
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace")


def sniff_sep(path: Path) -> str:
    sample = sample_text(path)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {",": first.count(","), "\t": first.count("\t"), "|": first.count("|"), ";": first.count(";")}
        best, count = max(counts.items(), key=lambda kv: kv[1])
        if count > 0:
            return best
        return r"\s+"


def read_header(path: Path) -> tuple[list[str], str, str, str]:
    sep = sniff_sep(path)
    engine = "python" if sep == r"\s+" else "c"
    last_error = ""
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df0 = pd.read_csv(path, sep=sep, engine=engine, encoding=enc, nrows=0)
            cols = [str(c) for c in df0.columns]
            if len(cols) > 1:
                return cols, sep, enc, "ok"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    for fallback_sep in [",", "\t", "|", ";", r"\s+"]:
        engine = "python" if fallback_sep == r"\s+" else "c"
        for enc in ("utf-8-sig", "latin1"):
            try:
                df0 = pd.read_csv(path, sep=fallback_sep, engine=engine, encoding=enc, nrows=0)
                cols = [str(c) for c in df0.columns]
                if len(cols) > 1:
                    return cols, fallback_sep, enc, "ok"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
    return [], sep, "unknown", last_error or "unable to parse header"


def count_rows_fast(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            count = 0
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return max(count - 1, 0)
    except Exception:
        return None


def role_scores(path: Path, columns: list[str]) -> tuple[str, dict[str, int], str]:
    fname = path.name.lower()
    ncols = {norm_col(c) for c in columns}
    scores = {role: 0 for role in BLS_ATUS_0324_FILES}

    if "sum" in fname or "summary" in fname:
        scores["activity_summary"] += 20
    if "who" in fname:
        scores["who"] += 20
    if "cps" in fname:
        scores["atus_cps"] += 20
    if "rost" in fname or "roster" in fname:
        scores["roster"] += 20
    if "resp" in fname or "respondent" in fname:
        scores["respondent"] += 20
    if ("act" in fname or "activity" in fname) and "cps" not in fname and "sum" not in fname and "summary" not in fname:
        scores["activity"] += 20

    if "TUCASEID" in ncols:
        for role in scores:
            scores[role] += 2

    activity_vars = {"TUACTIVITYN", "TRCODEP", "TRTIER1P", "TRTIER2P", "TRTIER3P", "TUSTARTTIM", "TUSTOPTIME", "TUACTDUR24"}
    respondent_vars = {"TEAGE", "TESEX", "PTDTRACE", "PEHSPNON", "TELFS", "TUYEAR", "TUDIARYDAY", "GEREG", "GTMETSTA"}
    cps_vars = {"PEMLR", "PESEX", "PRTAGE", "PEEDUCA", "PTDTRACE", "PEHSPNON", "GESTFIPS"}
    roster_vars = {"TULINENO", "TERELATE", "TEAGE", "TESEX"}
    who_vars = {"TUWHO_CODE", "TUWHO", "TUACTIVITYN"}

    scores["activity"] += 4 * len(activity_vars & ncols)
    scores["respondent"] += 3 * len(respondent_vars & ncols)
    scores["atus_cps"] += 3 * len(cps_vars & ncols)
    scores["roster"] += 3 * len(roster_vars & ncols)
    scores["who"] += 3 * len(who_vars & ncols)

    if "TUCASEID" in ncols and len(activity_vars & ncols) >= 2:
        scores["activity"] += 10
    if "TUCASEID" in ncols and len(respondent_vars & ncols) >= 3:
        scores["respondent"] += 8

    role = max(scores, key=lambda r: scores[r])
    if scores[role] <= 2:
        role = "unknown"
    notes = "; ".join([f"{k}={v}" for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if v])
    return role, scores, notes


def discover_data_files(raw_dir: Path, extract_dir: Path) -> pd.DataFrame:
    candidates: list[Path] = []
    for root in [raw_dir, extract_dir]:
        if root.exists():
            candidates.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".dat", ".txt"}])
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for path in sorted(candidates):
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        status = "ok"
        notes = ""
        columns: list[str] = []
        sep = ""
        enc = ""
        try:
            columns, sep, enc, header_status = read_header(path)
            if header_status != "ok":
                status = "header_failed"
                notes = header_status
        except Exception as exc:  # noqa: BLE001
            status = "header_failed"
            notes = f"{type(exc).__name__}: {exc}"
        n_rows = count_rows_fast(path)
        role, scores, score_notes = role_scores(path, columns)
        if notes:
            notes = notes + "; " + score_notes
        else:
            notes = score_notes
        records.append({
            "detected_role": role,
            "path": str(path),
            "file_size": path.stat().st_size,
            "n_rows": n_rows,
            "n_cols": len(columns),
            "columns_preview": "|".join(columns[:25]),
            "status": status,
            "notes": notes,
            "role_score": max(scores.values()) if scores else 0,
            "delimiter": sep,
            "encoding": enc,
        })
    if not records:
        return pd.DataFrame(columns=["detected_role", "path", "file_size", "n_rows", "n_cols", "columns_preview", "status", "notes", "role_score", "delimiter", "encoding"])
    return pd.DataFrame(records)


def choose_role_files(inventory: pd.DataFrame) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    if inventory.empty:
        return selected
    for role in BLS_ATUS_0324_FILES:
        subset = inventory[(inventory["detected_role"] == role) & (inventory["status"] == "ok") & (inventory["n_cols"] > 1)].copy()
        if subset.empty:
            continue
        subset["n_rows_sort"] = pd.to_numeric(subset["n_rows"], errors="coerce").fillna(-1)
        subset["file_size_sort"] = pd.to_numeric(subset["file_size"], errors="coerce").fillna(-1)
        subset = subset.sort_values(["role_score", "n_rows_sort", "file_size_sort"], ascending=[False, False, False])
        selected[role] = Path(str(subset.iloc[0]["path"]))
    return selected


def dtype_map_for_columns(columns: list[str]) -> dict[str, str]:
    keep_string_tokens = [
        "CASEID",
        "TRCODE",
        "TIER",
        "START",
        "STOP",
        "TIME",
        "ACTIVITY",
        "LINENO",
        "RACE",
        "HISP",
        "SEX",
        "LFS",
        "EMP",
    ]
    out: dict[str, str] = {}
    for col in columns:
        ncol = norm_col(col)
        if any(tok in ncol for tok in keep_string_tokens):
            out[col] = "string"
    return out


def read_atus_table(path: Path, role: str, fallback_cols: list[str] | None = None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info: dict[str, Any] = {"role": role, "path": str(path), "status": "not_read"}
    try:
        columns, sep, enc, header_status = read_header(path)
        if header_status != "ok":
            info.update({"status": "failed", "error": header_status})
            return None, info
        engine = "python" if sep == r"\s+" else "c"
        dtype_map = dtype_map_for_columns(columns)
        usecols: list[str] | None = None
        selected_column_read = False
        if fallback_cols:
            wanted = {norm_col(c) for c in fallback_cols}
            selected = [c for c in columns if norm_col(c) in wanted]
            if selected:
                usecols = selected
                selected_column_read = len(selected) < len(columns)
        dtype = dtype_map if usecols is None else {c: dtype_map.get(c, "string") for c in usecols if c in dtype_map}
        df = pd.read_csv(path, sep=sep, engine=engine, encoding=enc, dtype=dtype, usecols=usecols, low_memory=False)
        info.update({
            "status": "ok",
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "source_n_cols": int(len(columns)),
            "fallback": False,
            "selected_column_read": selected_column_read,
            "read_mode": "selected_columns" if selected_column_read else "full_columns",
            "selected_columns": list(df.columns) if selected_column_read else [],
        })
        return df, info
    except Exception as exc:  # noqa: BLE001
        info.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)})
        return None, info


def make_column_inventory_from_headers(selected_files: dict[str, Path]) -> pd.DataFrame:
    """Inventory every detected variable from selected source headers.

    The data tables are read with only inspection-needed columns for speed, so
    column inventory must come from headers rather than loaded DataFrame columns.
    """
    records: list[dict[str, Any]] = []
    for role in ["activity", "respondent", "atus_cps", "roster", "who", "activity_summary"]:
        path = selected_files.get(role)
        if path is None:
            continue
        columns, _sep, _enc, status = read_header(path)
        if status != "ok":
            records.append({
                "file_source": role,
                "variable_name": "",
                "normalized_variable_name": "",
                "inferred_role": "header_unavailable",
                "candidate_X": False,
                "candidate_Y": False,
                "candidate_G": False,
                "candidate_Z": False,
                "leakage_risk": "unknown",
                "notes": status,
            })
            continue
        for col in columns:
            records.append(classify_column_inventory(role, str(col)))
    return pd.DataFrame(records)


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def first_non_missing_by_case(df: pd.DataFrame | None, case_col: str | None, value_col: str | None) -> pd.Series | None:
    if df is None or case_col is None or value_col is None or case_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[case_col, value_col]].copy()
    tmp[case_col] = normalize_id_series(tmp[case_col])
    tmp[value_col] = tmp[value_col].astype("string").str.strip()
    tmp = tmp[tmp[case_col].notna() & tmp[value_col].notna() & tmp[value_col].ne("")]
    if tmp.empty:
        return None
    return tmp.groupby(case_col, sort=False)[value_col].first()


def code_digits(series: pd.Series, width: int = 6) -> pd.Series:
    s = series.astype("string").str.strip()
    digits = s.str.extract(r"(\d+)", expand=False)
    digits = digits.mask(digits.isna() | digits.eq(""), pd.NA)
    return digits.str.zfill(width)


def tier_digits(series: pd.Series, width: int) -> pd.Series:
    s = series.astype("string").str.strip()
    digits = s.str.extract(r"(\d+)", expand=False)
    digits = digits.mask(digits.isna() | digits.eq(""), pd.NA)
    return digits.str.zfill(width)


def labelize(prefix: str, values: pd.Series) -> pd.Series:
    values = values.astype("string")
    return values.mask(values.isna() | values.eq(""), pd.NA).map(lambda x: f"{prefix}_{x}" if pd.notna(x) else pd.NA)


def parse_hour(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour + minute / 60.0
    digits = re.sub(r"\D", "", text)
    if digits:
        if len(digits) in {3, 4}:
            hour = int(digits[:-2])
            minute = int(digits[-2:])
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour + minute / 60.0
        if len(digits) <= 2:
            hour = int(digits)
            if 0 <= hour <= 23:
                return float(hour)
    try:
        dt = pd.to_datetime(text, errors="coerce")
        if pd.notna(dt):
            return float(dt.hour) + float(dt.minute) / 60.0
    except Exception:
        return None
    return None


def daypart_from_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, dtype="string")
    hours = series.map(parse_hour)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[(hours >= 0) & (hours < 6)] = "night"
    out[(hours >= 6) & (hours < 12)] = "morning"
    out[(hours >= 12) & (hours < 18)] = "afternoon"
    out[(hours >= 18) & (hours < 24)] = "evening"
    return out.astype("string")


def map_sex(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"1", "1.0", "male", "m"}:
        return "male"
    if text in {"2", "2.0", "female", "f"}:
        return "female"
    if "male" in text and "female" not in text:
        return "male"
    if "female" in text:
        return "female"
    return pd.NA


def map_employment(value: Any) -> str:
    if pd.isna(value):
        return "unknown_or_other"
    text = str(value).strip().lower()
    numeric = re.sub(r"\.0$", "", text)
    if numeric in {"1", "2"}:
        return "employed"
    if numeric in {"3", "4"}:
        return "unemployed"
    if numeric in {"5", "6", "7"}:
        return "not_in_labor_force"
    if "unemploy" in text or "layoff" in text or "looking" in text:
        return "unemployed"
    if "not in labor" in text or "nilf" in text or "not_in_labor" in text:
        return "not_in_labor_force"
    if "employ" in text or "at work" in text or "absent" in text:
        return "employed"
    return "unknown_or_other"


def is_hispanic(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    numeric = re.sub(r"\.0$", "", text)
    if numeric == "1":
        return True
    if numeric == "2":
        return False
    if "non" in text and "hisp" in text:
        return False
    if "hisp" in text or "latino" in text:
        return True
    return None


def race_white_black(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    numeric = re.sub(r"\.0$", "", text)
    if numeric == "1":
        return "white"
    if numeric == "2":
        return "black"
    if "white" in text:
        return "white"
    if "black" in text or "african" in text:
        return "black"
    return None


def map_race_group(race_value: Any, hisp_value: Any, has_hispanic_col: bool) -> str | pd.NA:
    race = race_white_black(race_value)
    if race is None:
        return pd.NA
    if has_hispanic_col:
        hisp = is_hispanic(hisp_value)
        if hisp is True:
            return pd.NA
        if race == "white":
            return "non_hispanic_white"
        if race == "black":
            return "non_hispanic_black"
    if race == "white":
        return "white_ethnicity_unknown"
    if race == "black":
        return "black_ethnicity_unknown"
    return pd.NA


def derive_household_from_roster(roster: pd.DataFrame | None, case_col: str | None, age_col: str | None) -> tuple[pd.Series | None, str]:
    if roster is None or case_col is None or age_col is None:
        return None, "roster unavailable or missing age column"
    tmp = roster[[case_col, age_col]].copy()
    tmp[case_col] = normalize_id_series(tmp[case_col])
    tmp["age"] = pd.to_numeric(tmp[age_col], errors="coerce")
    tmp = tmp[tmp[case_col].notna() & tmp["age"].notna()]
    if tmp.empty:
        return None, "roster age values unavailable"
    flags = tmp.groupby(case_col)["age"].agg(
        has_under_6=lambda x: bool((x < 6).any()),
        has_6_17=lambda x: bool(((x >= 6) & (x <= 17)).any()),
    )
    out = pd.Series("no_child", index=flags.index, dtype="string")
    out[flags["has_6_17"]] = "has_child_6_17"
    out[flags["has_under_6"]] = "has_child_under_6"
    return out, "derived from roster household member age"


def make_splits(case_ids: pd.Series, seed: int) -> dict[str, str]:
    ids = pd.Series(case_ids.dropna().astype("string").unique()).sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(math.floor(n * SPLIT_FRACTIONS["train"]))
    n_cal = int(math.floor(n * SPLIT_FRACTIONS["calibration"]))
    n_repair = int(math.floor(n * SPLIT_FRACTIONS["repair"]))
    split_map: dict[str, str] = {}
    for split, arr in [
        ("train", ids[:n_train]),
        ("calibration", ids[n_train:n_train + n_cal]),
        ("repair", ids[n_train + n_cal:n_train + n_cal + n_repair]),
        ("test", ids[n_train + n_cal + n_repair:]),
    ]:
        for case_id in arr:
            split_map[str(case_id)] = split
    return split_map


def entropy_from_counts(counts: pd.Series | dict[str, int] | list[int]) -> tuple[float, float, float]:
    arr = np.asarray(list(counts.values()) if isinstance(counts, dict) else counts, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = arr[arr > 0] / total
    entropy = float(-(p * np.log(p)).sum())
    eff = float(math.exp(entropy))
    majority = float(p.max()) if len(p) else 0.0
    return entropy, eff, majority


def counts_dict(series: pd.Series, expected: list[str] | None = None) -> dict[str, int]:
    vc = series.dropna().astype("string").value_counts().to_dict()
    out = {str(k): int(v) for k, v in vc.items()}
    if expected:
        for label in expected:
            out.setdefault(str(label), 0)
        out = {k: out[k] for k in sorted(out)}
    return out


def build_episode_frame(
    activity: pd.DataFrame,
    respondent: pd.DataFrame,
    cps: pd.DataFrame | None,
    roster: pd.DataFrame | None,
    merge_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    act_case = find_col(activity.columns, ["TUCASEID"])
    resp_case = find_col(respondent.columns, ["TUCASEID"])
    if act_case is None or resp_case is None:
        raise ValueError("respondent and activity must both contain TUCASEID")

    activity = activity.copy()
    respondent = respondent.copy()
    activity[act_case] = normalize_id_series(activity[act_case])
    respondent[resp_case] = normalize_id_series(respondent[resp_case])

    code_col = find_col(activity.columns, ["TRCODEP"], contains_any=["TRCODE"])
    tier1_col = find_col(activity.columns, ["TRTIER1P", "TRTIER1", "TIER1"], contains_any=["TIER1"])
    tier2_col = find_col(activity.columns, ["TRTIER2P", "TRTIER2", "TIER2"], contains_any=["TIER2"])
    duration_col = find_col(activity.columns, ["TUACTDUR24"], contains_any=["DUR"])
    start_col = find_col(activity.columns, ["TUSTARTTIM", "TUSTARTTIME", "STARTTIME"], contains_any=["START"])
    stop_col = find_col(activity.columns, ["TUSTOPTIME", "TUSTOPTIM", "STOPTIME"], contains_any=["STOP"])
    where_col = find_col(activity.columns, ["TEWHERE"], contains_any=["WHERE"])
    activity_number_col = find_col(activity.columns, ["TUACTIVITY_N", "TUACTIVITYN"], contains_any=["ACTIVITYN"])

    y_notes: list[str] = []
    if tier1_col is not None:
        tier1 = tier_digits(activity[tier1_col], 2)
        y_notes.append(f"Y1 from {tier1_col}")
    elif code_col is not None:
        tier1 = code_digits(activity[code_col], 6).str.slice(0, 2)
        y_notes.append(f"Y1 inferred from first two digits of {code_col}")
    else:
        tier1 = pd.Series(pd.NA, index=activity.index, dtype="string")
        y_notes.append("Y1 unavailable: no tier1 or activity code")

    if tier2_col is not None:
        tier2 = tier_digits(activity[tier2_col], 4)
        y_notes.append(f"Y2 from {tier2_col}")
    elif code_col is not None:
        tier2 = code_digits(activity[code_col], 6).str.slice(0, 4)
        y_notes.append(f"Y2 inferred from first four digits of {code_col}")
    else:
        tier2 = pd.Series(pd.NA, index=activity.index, dtype="string")
        y_notes.append("Y2 unavailable: no tier2 or activity code")

    if code_col is not None:
        code_norm = code_digits(activity[code_col], 6)
        sleep_flag = code_norm.str.startswith("0101", na=False)
        sleep_rule = f"official/code-prefix sleep rule: {code_col} normalized to six digits and prefix 0101"
    elif tier1_col is not None:
        sleep_flag = tier_digits(activity[tier1_col], 2).eq("01")
        sleep_rule = "fallback broad rule: tier1 personal care/sleeping prefix 01, because precise activity code is unavailable"
    else:
        sleep_flag = pd.Series(False, index=activity.index)
        sleep_rule = "sleep cannot be precisely detected; no sleep exclusion applied"

    resp_sex_col = find_col(respondent.columns, ["TESEX", "PESEX", "SEX"], contains_any=["SEX"])
    resp_race_col = find_col(respondent.columns, ["PTDTRACE", "RACE", "TRACE"], contains_any=["RACE", "TRACE"])
    resp_hisp_col = find_col(respondent.columns, ["PEHSPNON", "HISP", "HISPANIC", "ETHNIC"], contains_any=["HISP", "ETHNIC"])
    resp_emp_col = find_col(respondent.columns, ["TELFS", "PEMLR", "EMPSTAT", "LABFORCE", "LFS"], contains_any=["TELFS", "PEMLR", "LFS", "LABOR", "EMP"])

    cps_case = find_col(cps.columns, ["TUCASEID"]) if cps is not None else None
    cps_sex_col = find_col(cps.columns, ["TESEX", "PESEX", "SEX"], contains_any=["SEX"]) if cps is not None else None
    cps_race_col = find_col(cps.columns, ["PTDTRACE", "RACE", "TRDTRACE"], contains_any=["RACE", "TRACE"]) if cps is not None else None
    cps_hisp_col = find_col(cps.columns, ["PEHSPNON", "HISP", "HISPANIC", "ETHNIC"], contains_any=["HISP", "ETHNIC"]) if cps is not None else None
    cps_emp_col = find_col(cps.columns, ["TELFS", "PEMLR", "EMPSTAT", "LABFORCE", "LFS"], contains_any=["TELFS", "PEMLR", "LFS", "LABOR", "EMP"]) if cps is not None else None

    sex_map = first_non_missing_by_case(respondent, resp_case, resp_sex_col)
    sex_source = f"respondent:{resp_sex_col}" if sex_map is not None else ""
    if sex_map is None:
        sex_map = first_non_missing_by_case(cps, cps_case, cps_sex_col)
        sex_source = f"atus_cps:{cps_sex_col}" if sex_map is not None else "unavailable"

    race_map = first_non_missing_by_case(respondent, resp_case, resp_race_col)
    race_source = f"respondent:{resp_race_col}" if race_map is not None else ""
    if race_map is None:
        race_map = first_non_missing_by_case(cps, cps_case, cps_race_col)
        race_source = f"atus_cps:{cps_race_col}" if race_map is not None else "unavailable"

    hisp_map = first_non_missing_by_case(respondent, resp_case, resp_hisp_col)
    hisp_source = f"respondent:{resp_hisp_col}" if hisp_map is not None else ""
    if hisp_map is None:
        hisp_map = first_non_missing_by_case(cps, cps_case, cps_hisp_col)
        hisp_source = f"atus_cps:{cps_hisp_col}" if hisp_map is not None else "unavailable"

    emp_map = first_non_missing_by_case(respondent, resp_case, resp_emp_col)
    emp_source = f"respondent:{resp_emp_col}" if emp_map is not None else ""
    if emp_map is None:
        emp_map = first_non_missing_by_case(cps, cps_case, cps_emp_col)
        emp_source = f"atus_cps:{cps_emp_col}" if emp_map is not None else "unavailable"

    roster_case = find_col(roster.columns, ["TUCASEID"]) if roster is not None else None
    roster_age_col = find_col(roster.columns, ["TEAGE", "PRTAGE", "AGE"], contains_any=["AGE"]) if roster is not None else None
    household_map, household_source = derive_household_from_roster(roster, roster_case, roster_age_col)

    case_ids = normalize_id_series(activity[act_case])
    episodes = pd.DataFrame({
        "TUCASEID": case_ids,
        "activity_row_number": np.arange(len(activity)),
        "y_tier1": labelize("tier1", tier1),
        "y_tier2_raw": labelize("tier2", tier2),
        "is_sleep": sleep_flag.astype(bool).to_numpy(),
    })
    if code_col is not None:
        episodes["activity_code_norm"] = code_digits(activity[code_col], 6)
    else:
        episodes["activity_code_norm"] = pd.NA
    if start_col is not None:
        episodes["daypart"] = daypart_from_series(activity[start_col]).to_numpy()
    else:
        episodes["daypart"] = pd.NA
    if sex_map is not None:
        episodes["sex_g"] = episodes["TUCASEID"].map(sex_map).map(map_sex).astype("string")
    else:
        episodes["sex_g"] = pd.NA
    if race_map is not None:
        race_raw = episodes["TUCASEID"].map(race_map)
        hisp_raw = episodes["TUCASEID"].map(hisp_map) if hisp_map is not None else pd.Series(pd.NA, index=episodes.index)
        episodes["race_g"] = [map_race_group(r, h, hisp_map is not None) for r, h in zip(race_raw, hisp_raw)]
        episodes["race_g"] = episodes["race_g"].astype("string")
    else:
        episodes["race_g"] = pd.NA
    if emp_map is not None:
        episodes["employment_z"] = episodes["TUCASEID"].map(emp_map).map(map_employment).astype("string")
    else:
        episodes["employment_z"] = "unknown_or_other"
    if household_map is not None:
        episodes["household_z"] = episodes["TUCASEID"].map(household_map).fillna("unknown_or_other").astype("string")
    else:
        episodes["household_z"] = pd.NA
    episodes["daypart_x_employment"] = (
        episodes["daypart"].astype("string").fillna("unknown_daypart")
        + "__"
        + episodes["employment_z"].astype("string").fillna("unknown_or_other")
    )
    episodes.loc[episodes["daypart"].isna() | episodes["employment_z"].isna(), "daypart_x_employment"] = pd.NA

    metadata = {
        "activity_columns": {
            "case": act_case,
            "activity_code": code_col,
            "tier1": tier1_col,
            "tier2": tier2_col,
            "duration": duration_col,
            "start": start_col,
            "stop": stop_col,
            "where": where_col,
            "activity_number": activity_number_col,
        },
        "respondent_context_columns": {
            "sex": sex_source,
            "race": race_source,
            "hispanic": hisp_source,
            "employment": emp_source,
            "household": household_source,
        },
        "label_notes": y_notes,
        "sleep_rule": sleep_rule,
        "hispanic_limitation": "Hispanic ethnicity unavailable; race groups are Black/White only" if hisp_map is None else "race groups restricted to non-Hispanic Black/White",
    }

    column_records = []
    for source_name, df in [("activity", activity), ("respondent", respondent), ("atus_cps", cps), ("roster", roster)]:
        if df is None:
            continue
        for col in df.columns:
            column_records.append(classify_column_inventory(source_name, str(col)))
    column_inventory = pd.DataFrame(column_records)
    merge_summary["constructed_metadata"] = metadata
    return episodes, metadata, column_inventory


def classify_column_inventory(source: str, col: str) -> dict[str, Any]:
    ncol = norm_col(col)
    lower = col.lower()
    y_cols = {"TRCODEP", "TRTIER1P", "TRTIER2P", "TRTIER3P"}
    duration_or_proxy = any(tok in ncol for tok in ["DUR", "STOP", "WHERE", "WHO", "SUMMARY", "ACTIVITYN"])
    activity_code_or_label = ncol in y_cols or "TRCODE" in ncol or "TIER" in ncol
    g_col = any(tok in ncol for tok in ["SEX", "RACE", "HISP", "ETHNIC", "TRACE"])
    z_col = any(tok in ncol for tok in ["START", "TELFS", "PEMLR", "LFS", "EMP", "CHILD", "AGE"])
    x_col = any(tok in ncol for tok in [
        "START",
        "DIARY",
        "DAY",
        "HOLIDAY",
        "YEAR",
        "AGE",
        "EDU",
        "EMP",
        "LFS",
        "MAR",
        "CHILD",
        "HH",
        "REG",
        "STATE",
        "METRO",
    ])
    leakage = activity_code_or_label or duration_or_proxy
    if g_col:
        x_col = False
    inferred: list[str] = []
    notes: list[str] = []
    if activity_code_or_label:
        inferred.append("activity label/code")
        notes.append("candidate Y source; exclude from future X")
    if "DUR" in ncol:
        inferred.append("duration")
        notes.append("activity proxy; exclude from primary X")
    if "STOP" in ncol:
        inferred.append("stop time")
        notes.append("post/start-derived activity proxy; exclude from primary X")
    if "WHERE" in ncol:
        inferred.append("where/location")
        notes.append("proxy risk; exclude from primary X")
    if "WHO" in ncol:
        inferred.append("who-with")
        notes.append("proxy risk; exclude from primary X")
    if g_col:
        inferred.append("demographic group")
        notes.append("candidate G; G variables cannot enter future X")
    if z_col:
        inferred.append("context/audit cell candidate")
    if x_col and not leakage:
        inferred.append("primary X candidate")
    if not inferred:
        inferred.append("unclassified")
    return {
        "file_source": source,
        "variable_name": col,
        "normalized_variable_name": ncol,
        "inferred_role": "; ".join(dict.fromkeys(inferred)),
        "candidate_X": bool(x_col and not leakage),
        "candidate_Y": bool(activity_code_or_label),
        "candidate_G": bool(g_col),
        "candidate_Z": bool(z_col),
        "leakage_risk": "high" if leakage else ("exclude_as_G" if g_col else "low_or_context_dependent"),
        "notes": "; ".join(dict.fromkeys(notes)),
    }


def attach_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    split_map = make_splits(out["TUCASEID"], seed)
    out["split"] = out["TUCASEID"].astype("string").map(split_map).astype("string")
    return out


def sample_one_episode_per_respondent(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    shuffled = df.sample(frac=1.0, random_state=seed)
    sampled = shuffled.drop_duplicates("TUCASEID", keep="first").sort_values("activity_row_number")
    return sampled.reset_index(drop=True)


def learn_topk_labels(df: pd.DataFrame, label_col: str, k: int) -> list[str]:
    train = df[df["split"].eq("train")]
    counts = train[label_col].dropna().astype("string").value_counts()
    return [str(x) for x in counts.head(k).index.tolist()]


def apply_topk(df: pd.DataFrame, source_col: str, labels: list[str], target_col: str) -> pd.DataFrame:
    out = df.copy()
    label_set = set(labels)
    out[target_col] = out[source_col].astype("string")
    out[target_col] = out[target_col].where(out[target_col].isin(label_set), "OTHER_ACTIVITY")
    out.loc[out[source_col].isna(), target_col] = pd.NA
    return out


def distribution_rows(
    df: pd.DataFrame,
    label_col: str,
    expected_labels: list[str] | None,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_values = ["all"] + SPLIT_ORDER
    if expected_labels is None:
        train = df[df["split"].eq("train")]
        expected = sorted(train[label_col].dropna().astype("string").unique().tolist())
    else:
        expected = list(expected_labels)
    for split in split_values:
        sub = df if split == "all" else df[df["split"].eq(split)]
        counts = counts_dict(sub[label_col], expected)
        entropy, eff, majority = entropy_from_counts(counts)
        total = int(sum(counts.values()))
        other_count = int(counts.get("OTHER_ACTIVITY", 0))
        zero_labels = [label for label in expected if counts.get(label, 0) == 0]
        for label, count in counts.items():
            rows.append({
                **metadata,
                "split": split,
                "label": label,
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
                "entropy": entropy,
                "effective_number_of_labels": eff,
                "majority_baseline": majority,
                "other_activity_fraction": float(other_count / total) if total else 0.0,
                "zero_support_labels": dumps(zero_labels),
                "has_zero_support_class": bool(zero_labels),
            })
    return rows


def support_rows_by_var(df: pd.DataFrame, var: str, unit: str, split_col: str = "split") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["all"] + SPLIT_ORDER:
        sub = df if split == "all" else df[df[split_col].eq(split)]
        total = len(sub)
        counts = sub[var].fillna("MISSING").astype("string").value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({
                "unit": unit,
                "split": split,
                "variable": var,
                "value": str(value),
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
                "n_respondents": int(sub.loc[sub[var].fillna("MISSING").astype("string").eq(str(value)), "TUCASEID"].nunique()) if total else 0,
            })
    return rows


def tv_distance(df: pd.DataFrame, g_col: str, z_col: str, expected_g: list[str] | None = None, expected_z: list[str] | None = None) -> tuple[float | None, str]:
    sub = df[[g_col, z_col]].dropna().copy()
    if sub.empty:
        return None, "no non-missing G/Z rows"
    groups = expected_g or sorted(sub[g_col].astype("string").unique().tolist())
    z_levels = expected_z or sorted(sub[z_col].astype("string").unique().tolist())
    if len(groups) < 2 or len(z_levels) == 0:
        return None, "less than two groups or no Z levels"
    probs: dict[str, np.ndarray] = {}
    for g in groups:
        vals = sub.loc[sub[g_col].astype("string").eq(g), z_col].astype("string")
        counts = vals.value_counts().reindex(z_levels, fill_value=0).to_numpy(dtype=float)
        total = counts.sum()
        if total <= 0:
            probs[g] = np.zeros(len(z_levels), dtype=float)
        else:
            probs[g] = counts / total
    best_tv = 0.0
    best_pair = ""
    for a, b in combinations(groups, 2):
        tv = 0.5 * float(np.abs(probs[a] - probs[b]).sum())
        if tv >= best_tv:
            best_tv = tv
            best_pair = f"{a} vs {b}"
    return best_tv, best_pair


def gz_table(df: pd.DataFrame, g_col: str, z_col: str, expected_g: list[str] | None = None, expected_z: list[str] | None = None) -> pd.DataFrame:
    sub = df[[g_col, z_col, "TUCASEID"]].dropna().copy()
    groups = expected_g or sorted(sub[g_col].astype("string").unique().tolist())
    z_levels = expected_z or sorted(sub[z_col].astype("string").unique().tolist())
    records: list[dict[str, Any]] = []
    counts = sub.groupby([g_col, z_col], dropna=False).agg(count=("TUCASEID", "size"), n_respondents=("TUCASEID", "nunique")).reset_index()
    lookup = {(str(r[g_col]), str(r[z_col])): (int(r["count"]), int(r["n_respondents"])) for _, r in counts.iterrows()}
    for g in groups:
        for z in z_levels:
            count, n_resp = lookup.get((str(g), str(z)), (0, 0))
            records.append({"g": str(g), "z": str(z), "count": count, "n_respondents": n_resp})
    return pd.DataFrame(records)


def gz_and_tv_stats(
    df: pd.DataFrame,
    g_col: str,
    z_col: str,
    expected_g: list[str] | None = None,
    expected_z: list[str] | None = None,
) -> tuple[pd.DataFrame, float | None, str, int]:
    sub = df[[g_col, z_col, "TUCASEID"]].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["g", "z", "count", "n_respondents"]), None, "no non-missing G/Z rows", 0
    groups = expected_g or sorted(sub[g_col].astype("string").unique().tolist())
    z_levels = expected_z or sorted(sub[z_col].astype("string").unique().tolist())
    if not groups or not z_levels:
        return pd.DataFrame(columns=["g", "z", "count", "n_respondents"]), None, "less than two groups or no Z levels", 0

    counts = sub.groupby([g_col, z_col], dropna=False).agg(count=("TUCASEID", "size"), n_respondents=("TUCASEID", "nunique")).reset_index()
    lookup = {(str(r[g_col]), str(r[z_col])): (int(r["count"]), int(r["n_respondents"])) for _, r in counts.iterrows()}
    records: list[dict[str, Any]] = []
    prob_by_group: dict[str, np.ndarray] = {}
    for g in groups:
        raw_counts: list[int] = []
        for z in z_levels:
            count, n_resp = lookup.get((str(g), str(z)), (0, 0))
            raw_counts.append(count)
            records.append({"g": str(g), "z": str(z), "count": count, "n_respondents": n_resp})
        arr = np.asarray(raw_counts, dtype=float)
        total = arr.sum()
        prob_by_group[str(g)] = arr / total if total > 0 else np.zeros(len(z_levels), dtype=float)
    gz = pd.DataFrame(records)
    min_cell = int(gz["count"].min()) if not gz.empty else 0
    if len(groups) < 2:
        return gz, None, "less than two groups or no Z levels", min_cell
    best_tv = 0.0
    best_pair = ""
    for a, b in combinations([str(x) for x in groups], 2):
        tv = 0.5 * float(np.abs(prob_by_group[a] - prob_by_group[b]).sum())
        if tv >= best_tv:
            best_tv = tv
            best_pair = f"{a} vs {b}"
    return gz, best_tv, best_pair, min_cell


def make_z_and_gz_outputs(unit_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    z_vars = {
        "daypart": "z_support_daypart.csv",
        "employment_z": "z_support_employment.csv",
        "household_z": "z_support_household.csv",
    }
    g_vars = {"sex_g": "sex", "race_g": "race"}
    z_support: dict[str, list[dict[str, Any]]] = {name: [] for name in z_vars}
    gz_rows: list[dict[str, Any]] = []
    tv_rows: list[dict[str, Any]] = []
    for unit, df in unit_frames.items():
        for split in ["all"] + SPLIT_ORDER:
            sub = df if split == "all" else df[df["split"].eq(split)]
            stats: dict[tuple[str, str], tuple[pd.DataFrame, float | None, str, int]] = {}
            for g_col, _g_name in g_vars.items():
                for z_col in ["daypart", "employment_z", "household_z", "daypart_x_employment"]:
                    stats[(g_col, z_col)] = gz_and_tv_stats(sub, g_col, z_col)
            for z_col, filename in z_vars.items():
                total = len(sub)
                z_counts = sub[z_col].fillna("MISSING").astype("string").value_counts(dropna=False)
                for z_value, count in z_counts.items():
                    row = {
                        "unit": unit,
                        "split": split,
                        "z_variable": z_col,
                        "z_value": str(z_value),
                        "count": int(count),
                        "fraction": float(count / total) if total else 0.0,
                        "n_respondents": int(sub.loc[sub[z_col].fillna("MISSING").astype("string").eq(str(z_value)), "TUCASEID"].nunique()) if total else 0,
                    }
                    for g_col, g_name in g_vars.items():
                        _gz, tv, pair, min_cell = stats[(g_col, z_col)]
                        row[f"min_gz_cell_{g_name}"] = min_cell
                        row[f"tv_given_{g_name}"] = tv
                        row[f"tv_pair_{g_name}"] = pair
                    z_support[z_col].append(row)
            for g_col, g_name in g_vars.items():
                for z_col in ["daypart", "employment_z", "household_z", "daypart_x_employment"]:
                    gz, tv, pair, min_cell = stats[(g_col, z_col)]
                    for _, r in gz.iterrows():
                        gz_rows.append({
                            "unit": unit,
                            "split": split,
                            "group_variable": g_col,
                            "group_name": g_name,
                            "z_variable": z_col,
                            "g_value": r["g"],
                            "z_value": r["z"],
                            "count": int(r["count"]),
                            "n_respondents": int(r["n_respondents"]),
                            "min_cell_count_for_design": min_cell,
                            "support_safe_ge_200": bool(min_cell >= 200),
                        })
                    tv_rows.append({
                        "unit": unit,
                        "split": split,
                        "group_variable": g_col,
                        "group_name": g_name,
                        "z_variable": z_col,
                        "tv_p_z_given_g": tv,
                        "tv_pair": pair,
                        "min_gz_cell_count": min_cell,
                        "support_safe_ge_200": bool(min_cell >= 200),
                        "n_z_levels": int(sub[z_col].dropna().astype("string").nunique()),
                        "n_groups": int(sub[g_col].dropna().astype("string").nunique()),
                    })
    z_dfs = {col: pd.DataFrame(rows) for col, rows in z_support.items()}
    return pd.DataFrame(gz_rows), pd.DataFrame(tv_rows), z_dfs


def make_split_support_summary(unit_frames: dict[str, pd.DataFrame], topk_by_unit: dict[str, dict[int, list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    g_vars = {"sex_g": "sex", "race_g": "race"}
    z_vars = ["daypart", "employment_z", "household_z", "daypart_x_employment"]
    y_specs: list[tuple[str, str, int | None]] = [("tier1", "y_tier1", None)]
    for k in TOPK_GRID:
        y_specs.append((f"tier2_top{k}", f"y_tier2_top{k}", k))

    for unit, base_df in unit_frames.items():
        df = base_df.copy()
        for k, labels in topk_by_unit.get(unit, {}).items():
            df = apply_topk(df, "y_tier2_raw", labels, f"y_tier2_top{k}")
        active_y_specs = [(name, col, k) for name, col, k in y_specs if col in df.columns]
        expected_y_by_col: dict[str, list[str]] = {}
        for _y_name, y_col, k in active_y_specs:
            if k is None:
                expected_y_by_col[y_col] = sorted(df[df["split"].eq("train")][y_col].dropna().astype("string").unique().tolist())
            else:
                expected_y_by_col[y_col] = topk_by_unit.get(unit, {}).get(k, []) + ["OTHER_ACTIVITY"]
        expected_g_by_col = {g_col: sorted(df[g_col].dropna().astype("string").unique().tolist()) for g_col in g_vars}
        expected_z_by_col = {z_col: sorted(df[z_col].dropna().astype("string").unique().tolist()) for z_col in z_vars}

        for split in SPLIT_ORDER:
            split_df = df[df["split"].eq(split)]
            for g_col, g_name in g_vars.items():
                expected_g = expected_g_by_col[g_col]
                for z_col in z_vars:
                    expected_z = expected_z_by_col[z_col]
                    design_base = split_df.dropna(subset=[g_col, z_col])
                    g_counts = counts_dict(design_base[g_col], expected_g)
                    z_counts = counts_dict(design_base[z_col], expected_z)
                    gz = gz_table(design_base, g_col, z_col, expected_g, expected_z)
                    min_gz = int(gz["count"].min()) if not gz.empty else 0
                    n_examples = int(len(design_base))
                    n_respondents = int(design_base["TUCASEID"].nunique())

                    for y_name, y_col, k in active_y_specs:
                        expected_y = expected_y_by_col[y_col]
                        y_counts = counts_dict(design_base[y_col], expected_y)
                        zero_y = [label for label, count in y_counts.items() if count == 0]
                        y_nonmissing = design_base[y_col].dropna().astype("string")
                        gzy_feasible = unit == "one_episode_per_respondent" or len(design_base) <= 500_000
                        if not gzy_feasible:
                            gzy_min = pd.NA
                            gzy_zero_cells = pd.NA
                            gzy_min_positive = pd.NA
                            gzy_note = "skipped for all_eligible_episodes because full GxZxY grid groupby over millions of episodes is not needed for inspection; primary one_episode unit is checked"
                        elif design_base.empty or not expected_y or not expected_g or not expected_z:
                            gzy_min = 0
                            gzy_zero_cells = len(expected_y) * len(expected_g) * len(expected_z)
                            gzy_min_positive = 0
                            gzy_note = ""
                        else:
                            gzy_counts = design_base.groupby([g_col, z_col, y_col]).size()
                            gzy_lookup = {(str(a), str(b), str(c)): int(v) for (a, b, c), v in gzy_counts.items()}
                            all_counts = [gzy_lookup.get((str(g), str(z), str(y)), 0) for g in expected_g for z in expected_z for y in expected_y]
                            positives = [x for x in all_counts if x > 0]
                            gzy_min = int(min(all_counts)) if all_counts else 0
                            gzy_zero_cells = int(sum(1 for x in all_counts if x == 0))
                            gzy_min_positive = int(min(positives)) if positives else 0
                            gzy_note = ""
                        entropy, eff, majority = entropy_from_counts(y_counts)
                        rows.append({
                            "candidate_design": f"unit={unit}|Y={y_name}|G={g_name}|Z={z_col}",
                            "unit": unit,
                            "split": split,
                            "y_design": y_name,
                            "top_k": k,
                            "group_design": g_name,
                            "z_design": z_col,
                            "n_respondents": n_respondents,
                            "n_examples": n_examples,
                            "y_n_labels_observed": int(y_nonmissing.nunique()),
                            "y_expected_labels": dumps(expected_y),
                            "y_counts_json": dumps(y_counts),
                            "y_min_expected_count": int(min(y_counts.values())) if y_counts else 0,
                            "y_entropy": entropy,
                            "y_effective_number_of_labels": eff,
                            "y_majority_baseline": majority,
                            "zero_support_y_labels": dumps(zero_y),
                            "has_zero_support_y": bool(zero_y),
                            "g_counts_json": dumps(g_counts),
                            "z_counts_json": dumps(z_counts),
                            "gz_min_cell_count": min_gz,
                            "gzy_summary_feasible": bool(gzy_feasible),
                            "gzy_min_cell_count_including_zeros": gzy_min,
                            "gzy_min_positive_cell_count": gzy_min_positive,
                            "gzy_zero_cell_count": gzy_zero_cells,
                            "gzy_note": gzy_note,
                        })
    return pd.DataFrame(rows)


def choose_recommendation(
    primary_df: pd.DataFrame,
    topk_by_unit: dict[str, dict[int, list[str]]],
    split_summary: pd.DataFrame,
    tv_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if primary_df.empty:
        return {
            "inspection_completed": False,
            "recommend_for_training": False,
            "reason": "No eligible non-sleep activity episodes.",
        }
    working = primary_df.copy()
    y_metrics: dict[int, dict[str, Any]] = {}
    for k, labels in topk_by_unit.get("one_episode_per_respondent", {}).items():
        working = apply_topk(working, "y_tier2_raw", labels, f"y_tier2_top{k}")
        train_labels = labels + ["OTHER_ACTIVITY"]
        per_split = distribution_rows(working, f"y_tier2_top{k}", train_labels, {"unit": "one_episode_per_respondent", "k": k})
        split_df = pd.DataFrame(per_split)
        split_agg = split_df.drop_duplicates(["split", "entropy"])[["split", "effective_number_of_labels", "majority_baseline", "other_activity_fraction", "has_zero_support_class"]]
        non_all = split_agg[split_agg["split"].isin(SPLIT_ORDER)]
        all_row = split_agg[split_agg["split"].eq("all")]
        y_metrics[k] = {
            "labels": train_labels,
            "min_effective_split": float(non_all["effective_number_of_labels"].min()) if not non_all.empty else 0.0,
            "max_majority_split": float(non_all["majority_baseline"].max()) if not non_all.empty else 1.0,
            "max_other_split": float(non_all["other_activity_fraction"].max()) if not non_all.empty else 1.0,
            "zero_support_any_split": bool(non_all["has_zero_support_class"].any()) if not non_all.empty else True,
            "effective_all": float(all_row["effective_number_of_labels"].iloc[0]) if not all_row.empty else 0.0,
            "majority_all": float(all_row["majority_baseline"].iloc[0]) if not all_row.empty else 1.0,
        }

    candidate_rows: list[dict[str, Any]] = []
    for k, ymet in y_metrics.items():
        for group_col, group_name in [("sex_g", "sex"), ("race_g", "race")]:
            for z_col in ["daypart", "employment_z", "household_z", "daypart_x_employment"]:
                split_design = split_summary[
                    split_summary["candidate_design"].eq(f"unit=one_episode_per_respondent|Y=tier2_top{k}|G={group_name}|Z={z_col}")
                ]
                if split_design.empty:
                    continue
                min_gz_split = int(split_design["gz_min_cell_count"].min())
                zero_y = bool(split_design["has_zero_support_y"].any())
                tv_row = tv_summary[
                    tv_summary["unit"].eq("one_episode_per_respondent")
                    & tv_summary["split"].eq("all")
                    & tv_summary["group_variable"].eq(group_col)
                    & tv_summary["z_variable"].eq(z_col)
                ]
                tv = float(tv_row["tv_p_z_given_g"].iloc[0]) if not tv_row.empty and pd.notna(tv_row["tv_p_z_given_g"].iloc[0]) else 0.0
                full_min_gz = int(tv_row["min_gz_cell_count"].iloc[0]) if not tv_row.empty else 0
                z_levels = int(tv_row["n_z_levels"].iloc[0]) if not tv_row.empty else 0
                g_support_min = int(min(counts_dict(primary_df[group_col]).values())) if primary_df[group_col].dropna().nunique() else 0
                y_ok = ymet["min_effective_split"] >= 6 and ymet["max_majority_split"] <= 0.55 and not ymet["zero_support_any_split"] and not zero_y
                support_ok = full_min_gz >= 200 and min_gz_split >= 20 and g_support_min >= 200 and 2 <= z_levels <= 16
                tv_ok = tv >= 0.05
                leakage_ok = z_col in {"daypart", "employment_z", "household_z", "daypart_x_employment"}
                score = (
                    ymet["min_effective_split"]
                    + 8.0 * tv
                    + math.log1p(max(full_min_gz, 0)) / 2.0
                    - 2.0 * ymet["max_majority_split"]
                    - (0.5 if z_col == "daypart_x_employment" else 0.0)
                )
                candidate_rows.append({
                    "k": k,
                    "group_col": group_col,
                    "group_name": group_name,
                    "z_col": z_col,
                    "score": score,
                    "y_ok": y_ok,
                    "support_ok": support_ok,
                    "tv_ok": tv_ok,
                    "leakage_ok": leakage_ok,
                    "recommend_ok": bool(y_ok and support_ok and tv_ok and leakage_ok),
                    "min_effective_split": ymet["min_effective_split"],
                    "max_majority_split": ymet["max_majority_split"],
                    "max_other_split": ymet["max_other_split"],
                    "zero_support_any_split": ymet["zero_support_any_split"] or zero_y,
                    "full_min_gz": full_min_gz,
                    "min_gz_split": min_gz_split,
                    "g_support_min": g_support_min,
                    "tv_p_z_given_g": tv,
                    "z_levels": z_levels,
                })
    if not candidate_rows:
        return {
            "inspection_completed": True,
            "recommend_for_training": False,
            "reason": "No evaluable candidate designs.",
            "constructed_metadata": metadata,
        }
    cand = pd.DataFrame(candidate_rows).sort_values(["recommend_ok", "score"], ascending=[False, False])
    best = cand.iloc[0].to_dict()
    recommend = bool(best["recommend_ok"])
    risk_bits: list[str] = []
    if not best["y_ok"]:
        risk_bits.append("Y support/entropy/majority/zero-support criteria are weak")
    if not best["support_ok"]:
        risk_bits.append("G x Z support is weak in one or more splits")
    if not best["tv_ok"]:
        risk_bits.append("TV(P(Z|G)) is below preferred 0.05 threshold")
    if best["z_col"] == "daypart_x_employment":
        risk_bits.append("daypart x employment is more fragmented than single-axis Z")
    if metadata.get("activity_columns", {}).get("activity_code") is None:
        risk_bits.append("sleep exclusion and labels rely on tier fallback instead of official activity code")
    if not risk_bits:
        risk_bits.append("largest risk is future leakage control: activity code/tier/duration/where/who/summary must remain excluded from X")
    return {
        "inspection_completed": True,
        "recommend_for_training": recommend,
        "recommended_y": {
            "name": "tier2_activity_topK_plus_OTHER_ACTIVITY",
            "top_k": int(best["k"]),
            "train_vocab": topk_by_unit.get("one_episode_per_respondent", {}).get(int(best["k"]), []),
            "min_effective_number_of_labels_by_split": float(best["min_effective_split"]),
            "max_majority_baseline_by_split": float(best["max_majority_split"]),
            "max_other_activity_fraction_by_split": float(best["max_other_split"]),
        },
        "recommended_g": {
            "name": best["group_name"],
            "column": best["group_col"],
            "min_group_support": int(best["g_support_min"]),
        },
        "recommended_z": {
            "name": best["z_col"].replace("_z", ""),
            "column": best["z_col"],
            "tv_p_z_given_g": float(best["tv_p_z_given_g"]),
            "full_gz_min_cell_count": int(best["full_min_gz"]),
            "min_gz_cell_count_across_splits": int(best["min_gz_split"]),
            "n_z_levels": int(best["z_levels"]),
        },
        "recommended_unit": "one_episode_per_respondent",
        "primary_unit_rationale": "one sampled eligible non-sleep activity episode per respondent keeps the sample unit close to independent respondents",
        "all_eligible_episodes_use": "robustness only; future training must use respondent-level splits and respondent-balanced weighting",
        "largest_risk": "; ".join(risk_bits),
        "candidate_rankings": candidate_rows,
        "constructed_metadata": metadata,
        "training_command": (
            "python ATUS/atus_activity_curve/run_atus_curve.py "
            "--data-dir ATUS/atus_activity_curve/data/extracted_0324 "
            f"--unit one_episode_per_respondent --y tier2_top{int(best['k'])} "
            f"--group {best['group_name']} --z {best['z_col'].replace('_z', '')} --seed 4"
        ),
    }


def write_group_support_outputs(output_dir: Path, unit_frames: dict[str, pd.DataFrame]) -> None:
    sex_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    for unit, df in unit_frames.items():
        sex_rows.extend(support_rows_by_var(df, "sex_g", unit))
        race_rows.extend(support_rows_by_var(df, "race_g", unit))
    write_csv(output_dir / "group_support_sex.csv", sex_rows)
    write_csv(output_dir / "group_support_race.csv", race_rows)


def write_activity_label_outputs(output_dir: Path, unit_frames: dict[str, pd.DataFrame], include_sleep_frames: dict[str, pd.DataFrame]) -> dict[str, dict[int, list[str]]]:
    tier1_rows: list[dict[str, Any]] = []
    tier2_rows: list[dict[str, Any]] = []
    topk_by_unit: dict[str, dict[int, list[str]]] = {}

    for sleep_policy, frames in [("exclude_sleep", unit_frames), ("include_sleep", include_sleep_frames)]:
        for unit, df in frames.items():
            if df.empty:
                continue
            tier1_rows.extend(distribution_rows(df, "y_tier1", None, {"unit": unit, "sleep_policy": sleep_policy, "label_design": "tier1", "top_k": None}))
            if sleep_policy == "exclude_sleep":
                topk_by_unit.setdefault(unit, {})
            for k in TOPK_GRID:
                labels = learn_topk_labels(df, "y_tier2_raw", k)
                if sleep_policy == "exclude_sleep":
                    topk_by_unit[unit][k] = labels
                mapped = apply_topk(df, "y_tier2_raw", labels, f"y_tier2_top{k}")
                expected = labels + ["OTHER_ACTIVITY"]
                tier2_rows.extend(distribution_rows(mapped, f"y_tier2_top{k}", expected, {"unit": unit, "sleep_policy": sleep_policy, "label_design": "tier2_topk_other", "top_k": k, "train_topk_vocab": dumps(labels)}))

    write_csv(output_dir / "activity_label_distribution_tier1.csv", tier1_rows)
    write_csv(output_dir / "activity_label_distribution_tier2_topk.csv", tier2_rows)
    return topk_by_unit


def respondent_activity_merge_summary(respondent: pd.DataFrame, activity: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    resp_case = find_col(respondent.columns, ["TUCASEID"])
    act_case = find_col(activity.columns, ["TUCASEID"])
    if resp_case is None or act_case is None:
        raise ValueError("Missing TUCASEID in respondent or activity")
    respondent = respondent.copy()
    activity = activity.copy()
    respondent[resp_case] = normalize_id_series(respondent[resp_case])
    activity[act_case] = normalize_id_series(activity[act_case])

    resp_unique = set(respondent[resp_case].dropna().astype(str))
    act_unique = set(activity[act_case].dropna().astype(str))
    intersection = resp_unique & act_unique
    summary = {
        "respondent_row_count": int(len(respondent)),
        "activity_row_count": int(len(activity)),
        "unique_respondents_in_respondent_file": int(len(resp_unique)),
        "unique_respondents_in_activity_file": int(len(act_unique)),
        "intersection_respondent_count": int(len(intersection)),
        "unmatched_activity_respondent_count": int(len(act_unique - resp_unique)),
        "unmatched_respondent_count": int(len(resp_unique - act_unique)),
        "respondent_duplicate_caseid_rows": int(len(respondent) - respondent[resp_case].nunique(dropna=True)),
        "activity_has_multiple_rows_per_caseid": bool(activity[act_case].value_counts().max() > 1),
        "respondent_one_row_per_caseid": bool(len(respondent) == respondent[resp_case].nunique(dropna=True)),
    }
    resp_one = respondent.drop_duplicates(resp_case, keep="first")
    merged = activity.merge(resp_one, left_on=act_case, right_on=resp_case, how="left", suffixes=("_activity", "_respondent"), indicator=True)
    merge_counts = merged["_merge"].value_counts().to_dict()
    summary["merge_indicator_counts"] = {str(k): int(v) for k, v in merge_counts.items()}
    return merged, summary, respondent


def write_filtering_summary(
    output_dir: Path,
    merge_summary: dict[str, Any],
    episodes_all: pd.DataFrame,
    eligible: pd.DataFrame,
    primary: pd.DataFrame,
    all_eligible: pd.DataFrame,
    metadata: dict[str, Any],
    read_infos: list[dict[str, Any]],
    selected_files: dict[str, Path],
) -> None:
    episode_counts = all_eligible.groupby("TUCASEID").size() if not all_eligible.empty else pd.Series(dtype=int)
    payload = {
        "inspection_completed": True,
        "selected_files": {role: str(path) for role, path in selected_files.items()},
        "read_status": read_infos,
        "merge_summary": merge_summary,
        "activity_total_rows": int(len(episodes_all)),
        "include_sleep_rows_with_labels": int(episodes_all.dropna(subset=["y_tier1", "y_tier2_raw"]).shape[0]),
        "sleep_episode_count": int(episodes_all["is_sleep"].sum()) if "is_sleep" in episodes_all else 0,
        "eligible_non_sleep_episode_count": int(len(eligible)),
        "eligible_non_sleep_respondent_count": int(eligible["TUCASEID"].nunique()),
        "one_episode_per_respondent_count": int(len(primary)),
        "all_eligible_episodes_count": int(len(all_eligible)),
        "all_eligible_episode_count_per_respondent": {
            "min": int(episode_counts.min()) if not episode_counts.empty else 0,
            "p25": float(episode_counts.quantile(0.25)) if not episode_counts.empty else 0.0,
            "median": float(episode_counts.median()) if not episode_counts.empty else 0.0,
            "mean": float(episode_counts.mean()) if not episode_counts.empty else 0.0,
            "p75": float(episode_counts.quantile(0.75)) if not episode_counts.empty else 0.0,
            "max": int(episode_counts.max()) if not episode_counts.empty else 0,
        },
        "constructed_metadata": metadata,
    }
    write_json(output_dir / "filtering_summary.json", payload)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def write_candidate_summary(
    output_dir: Path,
    recommendation: dict[str, Any],
    merge_summary: dict[str, Any],
    metadata: dict[str, Any],
    split_summary: pd.DataFrame,
    tv_summary: pd.DataFrame,
    download_success: bool,
    missing_downloads: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# ATUS Inspection-Only Candidate Design Summary")
    lines.append("")
    lines.append("No classifier, conformal prediction, IPW, or curve experiment was run.")
    lines.append("")
    lines.append("## Download And File Status")
    lines.append("")
    lines.append(f"- Automatic download success: `{download_success}`")
    lines.append(f"- Missing downloads after attempt: `{', '.join(missing_downloads) if missing_downloads else 'none'}`")
    lines.append(f"- Respondent rows: `{merge_summary.get('respondent_row_count', 'NA')}`")
    lines.append(f"- Activity rows: `{merge_summary.get('activity_row_count', 'NA')}`")
    lines.append(f"- Respondent/activity intersection: `{merge_summary.get('intersection_respondent_count', 'NA')}`")
    lines.append(f"- Unmatched activity respondents: `{merge_summary.get('unmatched_activity_respondent_count', 'NA')}`")
    lines.append(f"- Unmatched respondent records: `{merge_summary.get('unmatched_respondent_count', 'NA')}`")
    lines.append("")
    lines.append("## Label And Sleep Handling")
    lines.append("")
    lines.append(f"- Label notes: `{'; '.join(metadata.get('label_notes', []))}`")
    lines.append(f"- Sleep exclusion rule: `{metadata.get('sleep_rule', 'NA')}`")
    lines.append("- Main inspection uses non-sleep activity prediction. Include-sleep label distributions are written as robustness rows in the label distribution CSVs.")
    lines.append("")
    lines.append("## Leakage Rules For Future Training")
    lines.append("")
    lines.append("If Y is activity category, future X must exclude:")
    lines.append("")
    for item in [
        "activity code",
        "activity description",
        "tier activity labels",
        "duration",
        "stop time",
        "activity summary variables",
        "where/location variables in the primary design",
        "who-with variables in the primary design",
        "any activity-derived summary",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Primary X may include only start time or daypart source, diary day/weekend/holiday, year, age, education, employment status, marital status, household children/household size, region, and respondent/household pre-activity context.")
    lines.append("")
    lines.append("G variables cannot enter X. Y/Z true labels cannot enter propensity H. Future propensity H can only come from frozen classifier prediction/uncertainty profiles.")
    lines.append("")
    lines.append("## Recommendation Standard")
    lines.append("")
    for item in [
        "Y should have enough labels, preferably effective number of labels >= 6.",
        "Majority baseline should not be too high.",
        "Each respondent-level split should avoid zero-support Y classes.",
        "G support should be adequate.",
        "G x Z cells should be adequately supported; for one-episode primary, full-sample G x Z min cell count should preferably be >= 200.",
        "TV(P(Z|G)) should not be too small; TV >= 0.05 is preferred.",
        "Z should not be too fragmented.",
        "Leakage risk must be controllable.",
        "ATUS should be narratively distinct from BioBias and Folktables.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Recommended Design")
    lines.append("")
    lines.append(f"- Recommend ATUS for training: `{recommendation.get('recommend_for_training')}`")
    rec_y = recommendation.get("recommended_y", {})
    rec_g = recommendation.get("recommended_g", {})
    rec_z = recommendation.get("recommended_z", {})
    lines.append(f"- Recommended Y: `{rec_y.get('name', 'NA')}`")
    lines.append(f"- Recommended K: `{rec_y.get('top_k', 'NA')}`")
    lines.append(f"- Recommended G: `{rec_g.get('name', 'NA')}`")
    lines.append(f"- Recommended Z: `{rec_z.get('name', 'NA')}`")
    lines.append(f"- Recommended unit: `{recommendation.get('recommended_unit', 'NA')}`")
    lines.append(f"- Largest risk: `{recommendation.get('largest_risk', recommendation.get('reason', 'NA'))}`")
    lines.append(f"- Training command: `{recommendation.get('training_command', 'NA')}`")
    lines.append("")
    lines.append("## Top Candidate Metrics")
    lines.append("")
    cand = pd.DataFrame(recommendation.get("candidate_rankings", []))
    if not cand.empty:
        cand = cand.sort_values(["recommend_ok", "score"], ascending=[False, False]).head(12)
        rows = []
        for _, r in cand.iterrows():
            rows.append({
                "K": int(r["k"]),
                "G": r["group_name"],
                "Z": r["z_col"],
                "TV": round(float(r["tv_p_z_given_g"]), 4),
                "full_min_gz": int(r["full_min_gz"]),
                "split_min_gz": int(r["min_gz_split"]),
                "eff_min": round(float(r["min_effective_split"]), 2),
                "maj_max": round(float(r["max_majority_split"]), 3),
                "ok": bool(r["recommend_ok"]),
            })
        lines.append(md_table(rows, ["K", "G", "Z", "TV", "full_min_gz", "split_min_gz", "eff_min", "maj_max", "ok"]))
    else:
        lines.append("_No candidate metrics available._")
    lines.append("")
    lines.append("## Distinctness")
    lines.append("")
    lines.append("ATUS activity prediction uses respondent/time/household context, activity-category labels, demographic groups, and natural context cells such as daypart/employment/household context. This is substantively distinct from BioBias text/occupation prediction and Folktables ACS tabular tasks.")
    lines.append("")
    lines.append("Detailed CSV/JSON artifacts in this directory contain split support, G support, Z support, G x Z support, TV(P(Z|G)), and label entropy diagnostics.")
    (output_dir / "candidate_design_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placeholder_outputs(output_dir: Path, reason: str, inventory: pd.DataFrame, download_success: bool, missing_downloads: list[str], read_infos: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    column_inventory_columns = ["file_source", "variable_name", "normalized_variable_name", "inferred_role", "candidate_X", "candidate_Y", "candidate_G", "candidate_Z", "leakage_risk", "notes"]
    write_csv(output_dir / "column_inventory.csv", [{
        "file_source": "not_completed",
        "variable_name": "",
        "normalized_variable_name": "",
        "inferred_role": "unavailable",
        "candidate_X": False,
        "candidate_Y": False,
        "candidate_G": False,
        "candidate_Z": False,
        "leakage_risk": "unknown",
        "notes": reason,
    }], columns=column_inventory_columns)
    required_csvs = {
        "activity_label_distribution_tier1.csv": ["status", "reason"],
        "activity_label_distribution_tier2_topk.csv": ["status", "reason"],
        "split_support_summary.csv": ["status", "reason"],
        "group_support_sex.csv": ["status", "reason"],
        "group_support_race.csv": ["status", "reason"],
        "z_support_daypart.csv": ["status", "reason"],
        "z_support_employment.csv": ["status", "reason"],
        "z_support_household.csv": ["status", "reason"],
        "gz_support_summary.csv": ["status", "reason"],
        "tv_z_given_g_summary.csv": ["status", "reason"],
    }
    for filename, cols in required_csvs.items():
        write_csv(output_dir / filename, [{"status": "not_completed", "reason": reason}], columns=cols)
    if inventory is not None:
        write_csv(output_dir / "file_inventory.csv", inventory)
    payload = {
        "inspection_completed": False,
        "reason": reason,
        "download_success": download_success,
        "missing_downloads": missing_downloads,
        "read_status": read_infos or [],
    }
    write_json(output_dir / "filtering_summary.json", payload)
    rec = {
        "inspection_completed": False,
        "recommend_for_training": False,
        "reason": reason,
        "training_command": None,
    }
    write_json(output_dir / "recommended_atus_design.json", rec)
    write_candidate_summary(output_dir, rec, {}, {}, pd.DataFrame(), pd.DataFrame(), download_success, missing_downloads)
    return rec


def print_status(
    download_success: bool,
    missing_downloads: list[str],
    read_infos: list[dict[str, Any]],
    recommendation: dict[str, Any],
    merge_summary: dict[str, Any] | None = None,
    eligible_non_sleep_episode_count: int | None = None,
    eligible_non_sleep_one_episode_count: int | None = None,
) -> None:
    status_by_role = {info.get("role"): info.get("status") for info in read_infos}
    rec_y = recommendation.get("recommended_y", {})
    rec_g = recommendation.get("recommended_g", {})
    rec_z = recommendation.get("recommended_z", {})
    print("\n=== ATUS inspection status ===")
    print(f"1. automatic download success: {download_success}")
    print(f"2. manual download needed: {', '.join(missing_downloads) if missing_downloads else 'none'}")
    print(f"3. read status: respondent={status_by_role.get('respondent', 'missing')}, activity={status_by_role.get('activity', 'missing')}, atus_cps={status_by_role.get('atus_cps', 'missing')}, roster={status_by_role.get('roster', 'missing')}")
    print(f"4. respondent count: {(merge_summary or {}).get('respondent_row_count', 'NA')}")
    print(f"5. activity row count: {(merge_summary or {}).get('activity_row_count', 'NA')}")
    print(f"6. matched TUCASEID count: {(merge_summary or {}).get('intersection_respondent_count', 'NA')}")
    print(f"7. unmatched activity/respondent counts: {(merge_summary or {}).get('unmatched_activity_respondent_count', 'NA')} / {(merge_summary or {}).get('unmatched_respondent_count', 'NA')}")
    print(f"8. eligible non-sleep episode count: {eligible_non_sleep_episode_count if eligible_non_sleep_episode_count is not None else 'NA'}")
    print(f"9. eligible non-sleep one-episode count: {eligible_non_sleep_one_episode_count if eligible_non_sleep_one_episode_count is not None else 'NA'}")
    print(f"10. recommended Y: {rec_y.get('name', 'NA')}")
    print(f"11. recommended G: {rec_g.get('name', 'NA')}")
    print(f"12. recommended Z: {rec_z.get('name', 'NA')}")
    print(f"13. recommended unit: {recommendation.get('recommended_unit', 'NA')}")
    print(f"14. recommended K: {rec_y.get('top_k', 'NA')}")
    print(f"15. majority baseline: {rec_y.get('max_majority_baseline_by_split', 'NA')}")
    print(f"16. effective number of labels: {rec_y.get('min_effective_number_of_labels_by_split', 'NA')}")
    print(f"17. TV(P(Z|G)): {rec_z.get('tv_p_z_given_g', 'NA')}")
    print(f"18. min GxZ support: full={rec_z.get('full_gz_min_cell_count', 'NA')}, split={rec_z.get('min_gz_cell_count_across_splits', 'NA')}")
    print(f"19. recommend ATUS for training: {recommendation.get('recommend_for_training')}")
    print(f"20. largest risk: {recommendation.get('largest_risk', recommendation.get('reason', 'NA'))}")
    print(f"21. training command: {recommendation.get('training_command', 'NA')}")
    if missing_downloads:
        print("\nManual fallback: download the listed zip files from https://www.bls.gov/tus/datafiles.htm and place them in --data-dir, or place extracted csv/dat/txt files under --data-dir or --extract-dir, then rerun with --skip-download.")


def append_cmd_result(
    output_dir: Path,
    argv: list[str],
    recommendation: dict[str, Any],
    merge_summary: dict[str, Any] | None = None,
    eligible_non_sleep_episode_count: int | None = None,
    eligible_non_sleep_one_episode_count: int | None = None,
) -> None:
    rec_y = recommendation.get("recommended_y", {})
    rec_g = recommendation.get("recommended_g", {})
    rec_z = recommendation.get("recommended_z", {})
    cmd = " ".join([shlex_quote(sys.executable), *[shlex_quote(x) for x in argv]])
    lines = [
        "",
        "Run result summary:",
        "",
        "```bash",
        cmd,
        "```",
        "",
        f"- inspection_completed: `{recommendation.get('inspection_completed')}`",
        f"- respondent_count: `{(merge_summary or {}).get('respondent_row_count', 'NA')}`",
        f"- activity_row_count: `{(merge_summary or {}).get('activity_row_count', 'NA')}`",
        f"- matched_TUCASEID_count: `{(merge_summary or {}).get('intersection_respondent_count', 'NA')}`",
        f"- unmatched_activity_respondents: `{(merge_summary or {}).get('unmatched_activity_respondent_count', 'NA')}`",
        f"- unmatched_respondents: `{(merge_summary or {}).get('unmatched_respondent_count', 'NA')}`",
        f"- eligible_non_sleep_episode_count: `{eligible_non_sleep_episode_count if eligible_non_sleep_episode_count is not None else 'NA'}`",
        f"- eligible_non_sleep_one_episode_count: `{eligible_non_sleep_one_episode_count if eligible_non_sleep_one_episode_count is not None else 'NA'}`",
        f"- recommended_Y: `{rec_y.get('name', 'NA')}`",
        f"- recommended_K: `{rec_y.get('top_k', 'NA')}`",
        f"- recommended_G: `{rec_g.get('name', 'NA')}`",
        f"- recommended_Z: `{rec_z.get('name', 'NA')}`",
        f"- recommended_unit: `{recommendation.get('recommended_unit', 'NA')}`",
        f"- majority_baseline: `{rec_y.get('max_majority_baseline_by_split', 'NA')}`",
        f"- effective_number_of_labels: `{rec_y.get('min_effective_number_of_labels_by_split', 'NA')}`",
        f"- TV_P_Z_given_G: `{rec_z.get('tv_p_z_given_g', 'NA')}`",
        f"- min_GxZ_support_full: `{rec_z.get('full_gz_min_cell_count', 'NA')}`",
        f"- min_GxZ_support_split: `{rec_z.get('min_gz_cell_count_across_splits', 'NA')}`",
        f"- recommend_for_training: `{recommendation.get('recommend_for_training')}`",
        f"- largest_risk: `{recommendation.get('largest_risk', recommendation.get('reason', 'NA'))}`",
        "",
    ]
    with (output_dir / "cmd_inspect_atus.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    extract_dir = resolve_path(args.extract_dir)
    output_dir = resolve_path(args.output_dir)
    ensure_dirs(data_dir, extract_dir, output_dir)

    write_manual_download_instructions(output_dir)
    write_data_sources(output_dir)
    write_cmd_record(output_dir, sys.argv)

    download_log, missing_downloads, download_success = download_files(
        data_dir=data_dir,
        output_dir=output_dir,
        should_download=args.download,
        skip_download=args.skip_download,
    )
    extract_log = extract_archives(data_dir, extract_dir)
    if not extract_log.empty:
        extract_log_path = output_dir / "download_log.txt"
        with extract_log_path.open("a", encoding="utf-8") as f:
            f.write("\n\n# extraction_log\n")
            extract_log.to_csv(f, index=False)

    if missing_downloads:
        print("Automatic download did not provide all requested files.")
        print("Manual download needed for:")
        for filename in missing_downloads:
            print(f"  - {filename}")
        print(f"Place zip files in: {data_dir}")
        print(f"Or place extracted csv/dat/txt files in: {extract_dir}")
        print(f"Manual instructions written to: {output_dir / 'manual_download_instructions.md'}")

    inventory = discover_data_files(data_dir, extract_dir)
    write_csv(output_dir / "file_inventory.csv", inventory)
    selected_files = choose_role_files(inventory)
    missing_required_roles = sorted(REQUIRED_ROLES - set(selected_files))
    if missing_required_roles:
        reason = f"Cannot complete ATUS inspection: missing required detected files for {', '.join(missing_required_roles)}."
        rec = write_placeholder_outputs(output_dir, reason, inventory, download_success, missing_downloads)
        print_status(download_success, missing_downloads, [], rec)
        return 0

    read_infos: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame | None] = {}
    fallback_cols = [
        "TUCASEID", "TRCODEP", "TRTIER1P", "TRTIER2P", "TRTIER3P", "TUACTDUR24", "TUSTARTTIM", "TUSTOPTIME", "TEWHERE", "TUACTIVITY_N",
        "TEAGE", "TESEX", "PTDTRACE", "PEHSPNON", "TELFS", "TUYEAR", "TUDIARYDAY", "PEEDUCA", "TRCHILDNUM", "GEREG",
        "TULINENO", "TERELATE", "PEMLR", "PESEX", "PRTAGE",
    ]
    for role in ["respondent", "activity", "atus_cps", "roster"]:
        path = selected_files.get(role)
        if path is None:
            read_infos.append({"role": role, "status": "missing", "path": None})
            tables[role] = None
            continue
        df, info = read_atus_table(path, role, fallback_cols)
        read_infos.append(info)
        tables[role] = df

    respondent = tables.get("respondent")
    activity = tables.get("activity")
    if respondent is None or activity is None:
        reason = "Cannot complete ATUS inspection: respondent or activity file could not be read."
        rec = write_placeholder_outputs(output_dir, reason, inventory, download_success, missing_downloads, read_infos)
        print_status(download_success, missing_downloads, read_infos, rec)
        return 0

    try:
        _, merge_summary, respondent_one = respondent_activity_merge_summary(respondent, activity)
        episodes_all, metadata, _loaded_column_inventory = build_episode_frame(
            activity=activity,
            respondent=respondent_one,
            cps=tables.get("atus_cps"),
            roster=tables.get("roster"),
            merge_summary=merge_summary,
        )
        column_inventory = make_column_inventory_from_headers(selected_files)
    except Exception as exc:  # noqa: BLE001
        reason = f"Cannot complete ATUS inspection after reading files: {type(exc).__name__}: {exc}"
        rec = write_placeholder_outputs(output_dir, reason, inventory, download_success, missing_downloads, read_infos)
        print_status(download_success, missing_downloads, read_infos, rec)
        return 0

    write_csv(output_dir / "column_inventory.csv", column_inventory, columns=[
        "file_source",
        "variable_name",
        "normalized_variable_name",
        "inferred_role",
        "candidate_X",
        "candidate_Y",
        "candidate_G",
        "candidate_Z",
        "leakage_risk",
        "notes",
    ])

    include_sleep = episodes_all.dropna(subset=["y_tier1", "y_tier2_raw"]).copy()
    eligible = episodes_all[~episodes_all["is_sleep"]].dropna(subset=["y_tier1", "y_tier2_raw"]).copy()
    include_sleep_primary = attach_split(sample_one_episode_per_respondent(include_sleep, args.seed), args.seed)
    include_sleep_all = attach_split(include_sleep.reset_index(drop=True), args.seed)
    primary = attach_split(sample_one_episode_per_respondent(eligible, args.seed), args.seed)
    all_eligible = attach_split(eligible.reset_index(drop=True), args.seed)

    unit_frames = {
        "one_episode_per_respondent": primary,
        "all_eligible_episodes": all_eligible,
    }
    include_sleep_frames = {
        "one_episode_per_respondent": include_sleep_primary,
        "all_eligible_episodes": include_sleep_all,
    }

    topk_by_unit = write_activity_label_outputs(output_dir, unit_frames, include_sleep_frames)
    write_group_support_outputs(output_dir, unit_frames)

    gz_summary, tv_summary, z_dfs = make_z_and_gz_outputs(unit_frames)
    write_csv(output_dir / "gz_support_summary.csv", gz_summary)
    write_csv(output_dir / "tv_z_given_g_summary.csv", tv_summary)
    write_csv(output_dir / "z_support_daypart.csv", z_dfs["daypart"])
    write_csv(output_dir / "z_support_employment.csv", z_dfs["employment_z"])
    write_csv(output_dir / "z_support_household.csv", z_dfs["household_z"])

    split_summary = make_split_support_summary(unit_frames, topk_by_unit)
    write_csv(output_dir / "split_support_summary.csv", split_summary)

    write_filtering_summary(
        output_dir=output_dir,
        merge_summary=merge_summary,
        episodes_all=episodes_all,
        eligible=eligible,
        primary=primary,
        all_eligible=all_eligible,
        metadata=metadata,
        read_infos=read_infos,
        selected_files=selected_files,
    )

    recommendation = choose_recommendation(primary, topk_by_unit, split_summary, tv_summary, metadata)
    write_json(output_dir / "recommended_atus_design.json", recommendation)
    write_candidate_summary(output_dir, recommendation, merge_summary, metadata, split_summary, tv_summary, download_success, missing_downloads)
    append_cmd_result(
        output_dir,
        sys.argv,
        recommendation,
        merge_summary=merge_summary,
        eligible_non_sleep_episode_count=int(len(eligible)),
        eligible_non_sleep_one_episode_count=int(len(primary)),
    )

    print_status(
        download_success,
        missing_downloads,
        read_infos,
        recommendation,
        merge_summary=merge_summary,
        eligible_non_sleep_episode_count=int(len(eligible)),
        eligible_non_sleep_one_episode_count=int(len(primary)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
