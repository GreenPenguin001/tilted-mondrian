#!/usr/bin/env python3
"""
Stage 2: repair-only BioBias curve, Simpson diagnostics, and product uncertainty diagnostics.

All thresholds, aggregate gaps, cell gaps, active Simpson cells, and sample traces
are computed from D_repair only. This script intentionally does not read D_cal,
so its active_simpson_curve_cells.csv can be safely used by a downstream
repair algorithm.

The script consumes probability artifacts produced by biobias_train_export.py
and generates the following diagnostic figures:

1. Main figure per score:
   aggregate curve + Simpson evidence map + persistence + strongest witness.
2. Occupation curve grid per score:
   2x5 occupation-conditioned curves with aggregate references.
3. Global support-count and occupation-composition sanity figures.
4. Global score-robustness heatmap when multiple scores are analyzed.
5. Product-type uncertainty diagnostic per score.
6. Mechanism diagnostics linking Simpson cells to entropy and product slack.

No model training is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BIOBIAS_PROFESSION_ID_TO_NAME = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher",
}

BIOBIAS_GENDER_ID_TO_NAME = {0: "Male", 1: "Female"}


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def jload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jdump(obj: Any, path: Path) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [convert(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    path.write_text(json.dumps(convert(obj), indent=2), encoding="utf-8")



def parse_grid(text: str) -> list[float]:
    presets = {
        "dense_safe": np.round(np.arange(0.02, 0.1001, 0.005), 3).tolist(),
        "dense_010": np.round(np.arange(0.02, 0.1001, 0.005), 3).tolist(),
        "dense_012": np.round(np.arange(0.02, 0.1201, 0.005), 3).tolist(),
    }
    key = str(text).strip().lower()
    if key in presets:
        values = presets[key]
    else:
        values = sorted({float(x) for x in str(text).split(",") if x.strip()})
    if not values or any(x <= 0.0 or x >= 1.0 for x in values):
        raise ValueError("--alpha-grid must contain values in (0, 1), or use a preset: dense_safe, dense_010, dense_012.")
    return [float(x) for x in values]

def is_numeric_string(x: Any) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False


def is_placeholder_name(x: Any) -> bool:
    s = str(x).strip()
    if not s:
        return True
    if is_numeric_string(s):
        return True
    return s.startswith("label_") or s.startswith("occupation id ") or s.startswith("profession_")


def canonical_profession_name(profession_id: int, fallback: Any | None = None) -> str:
    if fallback is not None and not is_placeholder_name(fallback):
        return str(fallback)
    if int(profession_id) in BIOBIAS_PROFESSION_ID_TO_NAME:
        return BIOBIAS_PROFESSION_ID_TO_NAME[int(profession_id)]
    return f"profession_{int(profession_id)}"


def clean_label(x: object) -> str:
    s = str(x).strip()
    if not s:
        return s
    try:
        v = float(s)
        if v.is_integer() and int(v) in BIOBIAS_PROFESSION_ID_TO_NAME:
            return BIOBIAS_PROFESSION_ID_TO_NAME[int(v)]
    except Exception:
        pass
    return s.replace("_", " ")


def default_meta(k: int) -> dict[str, Any]:
    return {
        "label_names": [f"label_{i}" for i in range(k)],
        "occupation_names": [f"label_{i}" for i in range(k)],
        "audit_names": [f"label_{i}" for i in range(k)],
        "gender_names": ["Male", "Female"],
        "group_names": ["Male", "Female"],
        "num_labels": int(k),
        "meta_source": "fallback_meta_without_label_map",
    }


def label_maps_to_meta(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    selected = payload.get("selected_professions") or []
    raw_names = payload.get("label_names") or payload.get("occupation_names") or payload.get("audit_names") or []
    raw_pid = payload.get("compact_label_to_profession_id") or payload.get("new_label_to_old_profession_id") or {}
    raw_name = payload.get("compact_label_to_occupation_name") or {}
    pid_to_name = {str(k): str(v) for k, v in BIOBIAS_PROFESSION_ID_TO_NAME.items()}
    pid_to_name.update({str(k): str(v) for k, v in payload.get("profession_id_to_name", {}).items()})

    if selected:
        compact_to_pid = {int(r["compact_label"]): int(r["profession_id"]) for r in selected}
        compact_to_name = {int(r["compact_label"]): str(r.get("occupation_name", "")) for r in selected}
    else:
        compact_to_pid = {int(k): int(v) for k, v in raw_pid.items()}
        compact_to_name = {int(k): str(v) for k, v in raw_name.items()}

    if not compact_to_pid:
        compact_to_pid = {i: i for i in range(len(raw_names))}

    k = int(payload.get("num_labels", max(compact_to_pid.keys()) + 1 if compact_to_pid else len(raw_names)))
    selected_out = []
    labels = []
    for compact in range(k):
        pid = int(compact_to_pid.get(compact, compact))
        fallback = compact_to_name.get(compact, raw_names[compact] if compact < len(raw_names) else None)
        if is_placeholder_name(fallback):
            fallback = pid_to_name.get(str(pid))
        name = canonical_profession_name(pid, fallback)
        labels.append(name)
        selected_out.append({
            "compact_label": int(compact),
            "profession_id": int(pid),
            "occupation_name": name,
        })

    gender_names = payload.get("gender_names") or payload.get("group_names") or []
    if not gender_names or all(is_placeholder_name(x) for x in gender_names[:2]):
        gender_names = [BIOBIAS_GENDER_ID_TO_NAME[0], BIOBIAS_GENDER_ID_TO_NAME[1]]
    gender_names = [str(x) for x in gender_names]

    return {
        "label_names": labels,
        "occupation_names": labels,
        "audit_names": labels,
        "gender_names": gender_names,
        "group_names": gender_names,
        "num_labels": int(k),
        "selected_professions": selected_out,
        "compact_label_to_profession_id": {str(r["compact_label"]): int(r["profession_id"]) for r in selected_out},
        "compact_label_to_occupation_name": {str(r["compact_label"]): str(r["occupation_name"]) for r in selected_out},
        "profession_id_to_name": pid_to_name,
        "label_maps": payload,
        "meta_source": str(source),
    }


def normalize_meta_payload(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    if payload.get("schema_version", "").startswith("biobias_label_maps"):
        return label_maps_to_meta(payload, source)
    if "label_maps" in payload:
        lm_meta = label_maps_to_meta(payload["label_maps"], source)
        base = dict(payload.get("meta", {}))
        base.update(lm_meta)
        base["meta_source"] = str(source)
        return base
    if "meta" in payload:
        meta = dict(payload["meta"])
        if "label_names" in meta and "old_profession_ids" in meta:
            label_map_payload = {
                "label_names": meta.get("label_names", []),
                "new_label_to_old_profession_id": {str(i): int(pid) for i, pid in enumerate(meta.get("old_profession_ids", []))},
                "gender_names": meta.get("gender_names", []),
                "num_labels": int(meta.get("num_labels", len(meta.get("label_names", [])))),
            }
            meta.update(label_maps_to_meta(label_map_payload, source))
        meta["meta_source"] = str(source)
        return meta
    if any(k in payload for k in ["label_names", "occupation_names", "audit_names", "new_label_to_old_profession_id", "compact_label_to_profession_id"]):
        return label_maps_to_meta(payload, source)
    meta = dict(payload)
    meta["meta_source"] = str(source)
    return meta


def override_names_from_cli(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    meta = dict(meta)
    if args.label_names:
        labels = [x.strip() for x in args.label_names.split(",") if x.strip()]
        if labels:
            meta["label_names"] = labels
            meta["occupation_names"] = labels
            meta["audit_names"] = labels
            meta["num_labels"] = len(labels)
            meta["meta_source"] = str(meta.get("meta_source", "")) + ";cli_label_names"
    if args.group_names:
        groups = [x.strip() for x in args.group_names.split(",") if x.strip()]
        if groups:
            meta["gender_names"] = groups
            meta["group_names"] = groups
            meta["meta_source"] = str(meta.get("meta_source", "")) + ";cli_group_names"
    return meta


def load_meta(prob_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    candidates: list[Path] = []
    if args.meta_json is not None:
        candidates.append(args.meta_json.expanduser().resolve())
    candidates += [
        prob_dir / "label_maps.json",
        prob_dir / "split_meta.json",
        prob_dir / "meta.json",
        prob_dir.parent / "label_maps.json",
        prob_dir.parent / "split_meta.json",
        prob_dir.parent / "meta.json",
        prob_dir.parent.parent / "split_meta.json",
    ]
    for path in candidates:
        if path.exists():
            return override_names_from_cli(normalize_meta_payload(jload(path), path), args)
    k = int(np.load(prob_dir / "probs_repair.npy", mmap_mode="r").shape[1])
    return override_names_from_cli(default_meta(k), args)


def resolve_seed_dir(path: Path | None, run_root: Path | None, seed: int | None) -> Path | None:
    base = path if path is not None else run_root
    if base is None:
        return None
    base = base.expanduser().resolve()
    if (base / "probabilities").exists():
        return base
    if seed is not None and (base / f"seed_{seed}" / "probabilities").exists():
        return base / f"seed_{seed}"
    hits = sorted(base.glob("seed_*/probabilities"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError("Use --seed-dir, --run-root with --seed, --prob-dir, or --prob-zip.")


def resolve_probability_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.prob_zip is not None:
        tmp = Path(tempfile.mkdtemp(prefix="biobias_probabilities_"))
        with zipfile.ZipFile(args.prob_zip.expanduser().resolve()) as zf:
            zf.extractall(tmp)
        candidates = list(tmp.rglob("probs_repair.npy"))
        if not candidates:
            shutil.rmtree(tmp, ignore_errors=True)
            raise FileNotFoundError("Could not find probs_repair.npy inside --prob-zip.")
        return candidates[0].parent, tmp
    if args.prob_dir is not None:
        return args.prob_dir.expanduser().resolve(), None
    seed_dir = resolve_seed_dir(args.seed_dir, args.run_root, args.seed)
    if seed_dir is None:
        raise FileNotFoundError("No input artifact specified.")
    return seed_dir / "probabilities", None


def load_split(prob_dir: Path, split: str) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {
        "probs": np.load(prob_dir / f"probs_{split}.npy"),
        "y": np.load(prob_dir / f"y_{split}.npy"),
        "g": np.load(prob_dir / f"g_{split}.npy"),
        "z": np.load(prob_dir / f"z_{split}.npy"),
    }
    for key in ["idx", "artifact_index", "sample_id", "source_split", "source_index", "profession_id", "occupation_name", "group_name", "z_profession_id", "z_occupation_name"]:
        path = prob_dir / f"{key}_{split}.npy"
        if path.exists():
            data[key] = np.load(path, allow_pickle=True)
    return data


def load_artifact(args: argparse.Namespace) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], Path, Path | None]:
    prob_dir, tmp_dir = resolve_probability_dir(args)
    missing = [name for name in ["probs_repair.npy"] if not (prob_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {prob_dir}: {missing}")
    data = {"repair": load_split(prob_dir, "repair")}
    meta = load_meta(prob_dir, args)
    return data, meta, prob_dir, tmp_dir


def group_name(meta: dict[str, Any], group: int) -> str:
    names = meta.get("gender_names", []) or meta.get("group_names", [])
    if 0 <= int(group) < len(names):
        return str(names[int(group)])
    return BIOBIAS_GENDER_ID_TO_NAME.get(int(group), f"group_{int(group)}")


def audit_profession_id(meta: dict[str, Any], audit: int) -> int:
    mapping = meta.get("compact_label_to_profession_id", {}) or meta.get("new_label_to_old_profession_id", {})
    return int(mapping.get(str(int(audit)), mapping.get(int(audit), int(audit))))


def audit_name(meta: dict[str, Any], audit: int) -> str:
    names = meta.get("occupation_names", []) or meta.get("label_names", []) or meta.get("audit_names", [])
    pid = audit_profession_id(meta, audit)
    if 0 <= int(audit) < len(names) and not is_placeholder_name(names[int(audit)]):
        return str(names[int(audit)])
    pid_to_name = meta.get("profession_id_to_name", {})
    return canonical_profession_name(pid, pid_to_name.get(str(pid)))


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    if abs(float(temperature) - 1.0) > 1e-12:
        p = p ** (1.0 / float(temperature))
    return p / p.sum(axis=1, keepdims=True)


def scores_from_probs(
    probs: np.ndarray,
    score: str,
    saps_lambda: float,
    raps_lambda: float,
    raps_k_reg: int,
    randomize: bool,
    seed: int,
    temperature: float,
) -> np.ndarray:
    p = apply_temperature(probs, temperature)
    if score == "simple":
        return 1.0 - p

    n, k = p.shape
    u = (np.random.default_rng(seed).random(n) if randomize else np.full(n, 0.5))[:, None]
    order = np.argsort(-p, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(1, k + 1)
    p_sorted = np.take_along_axis(p, order, axis=1)
    before_sorted = np.c_[np.zeros(n), np.cumsum(p_sorted, axis=1)[:, :-1]]
    before = before_sorted[np.arange(n)[:, None], rank - 1]

    if score == "aps":
        return before + 0.5 * p
    if score == "raps":
        return before + u * p + float(raps_lambda) * np.maximum(rank - int(raps_k_reg), 0)
    if score == "saps":
        return np.where(rank == 1, u * p, p.max(axis=1, keepdims=True) + float(saps_lambda) * (rank - 2 + u))
    raise ValueError(f"Unknown score: {score}")


def true_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return scores[np.arange(len(y)), y]


def qhat(values: np.ndarray, alpha: float) -> float:
    s = np.asarray(values, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    return float("inf") if k > n else float(np.partition(s, k - 1)[k - 1])


def group_qhat(values: np.ndarray, groups: np.ndarray, alpha: float) -> dict[int, float]:
    return {int(g): qhat(values[groups == g], alpha) for g in sorted(np.unique(groups))}


def thresholds_for(groups: np.ndarray, threshold: float | dict[int, float]) -> np.ndarray:
    if isinstance(threshold, dict):
        return np.asarray([threshold[int(g)] for g in groups], dtype=float)
    return np.full(len(groups), float(threshold))


def threshold_value(threshold: float | dict[int, float], group: int) -> float:
    if isinstance(threshold, dict):
        return float(threshold[int(group)])
    return float(threshold)


def set_sizes(scores: np.ndarray, groups: np.ndarray, threshold: float | dict[int, float]) -> np.ndarray:
    thr = thresholds_for(groups, threshold)
    return (scores <= thr[:, None]).sum(axis=1)


def covered(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, threshold: float | dict[int, float]) -> np.ndarray:
    thr = thresholds_for(groups, threshold)
    return true_scores(scores, y) <= thr


def build_thresholds(cal: dict[str, np.ndarray], alphas: list[float]) -> dict[str, dict[float, float | dict[int, float]]]:
    ts = true_scores(cal["S"], cal["y"])
    return {
        "pooled": {alpha: qhat(ts, alpha) for alpha in alphas},
        "groupwise": {alpha: group_qhat(ts, cal["g"], alpha) for alpha in alphas},
    }


def curve_table(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    rows = []
    for policy in ["pooled", "groupwise"]:
        for alpha in alphas:
            threshold = thresholds[policy][alpha]
            size = set_sizes(split["S"], split["g"], threshold)
            cov = covered(split["S"], split["y"], split["g"], threshold)
            for group in sorted(np.unique(split["g"])):
                mask = split["g"] == group
                rows.append({
                    "split": split_name,
                    "alpha": float(alpha),
                    "policy": policy,
                    "group": int(group),
                    "group_name": group_name(meta, int(group)),
                    "n": int(mask.sum()),
                    "avg_set_size": float(size[mask].mean()),
                    "coverage": float(cov[mask].mean()),
                    "threshold": threshold_value(threshold, int(group)),
                })
    return pd.DataFrame(rows)


def aggregate_gap_table(curve: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(curve["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = map(int, groups)
    rows = []
    for (split, policy, alpha), sub in curve.groupby(["split", "policy", "alpha"]):
        sa = sub[sub.group == a].iloc[0]
        sb = sub[sub.group == b].iloc[0]
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "group_a": a,
            "group_a_name": str(sa.group_name),
            "group_b": b,
            "group_b_name": str(sb.group_name),
            "delta_V_a_minus_b": float(sa.avg_set_size - sb.avg_set_size),
            "delta_C_a_minus_b": float(sa.coverage - sb.coverage),
            "V_a": float(sa.avg_set_size),
            "V_b": float(sb.avg_set_size),
            "C_a": float(sa.coverage),
            "C_b": float(sb.coverage),
        })
    return pd.DataFrame(rows)


def cell_curve_table(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    split_name: str,
    policy: str,
) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        threshold = thresholds[policy][alpha]
        size = set_sizes(split["S"], split["g"], threshold)
        cov = covered(split["S"], split["y"], split["g"], threshold)
        for group in sorted(np.unique(split["g"])):
            for audit in sorted(np.unique(split["z"])):
                mask = (split["g"] == group) & (split["z"] == audit)
                if not mask.any():
                    continue
                rows.append({
                    "split": split_name,
                    "alpha": float(alpha),
                    "policy": policy,
                    "group": int(group),
                    "group_name": group_name(meta, int(group)),
                    "audit": int(audit),
                    "profession_id": audit_profession_id(meta, int(audit)),
                    "occupation_name": audit_name(meta, int(audit)),
                    "audit_name": audit_name(meta, int(audit)),
                    "n": int(mask.sum()),
                    "avg_set_size": float(size[mask].mean()),
                    "coverage": float(cov[mask].mean()),
                    "threshold": threshold_value(threshold, int(group)),
                })
    return pd.DataFrame(rows)


def simpson_gap_curve(
    aggregate_gaps: pd.DataFrame,
    cell_curve: pd.DataFrame,
    n_min: int,
    tau_agg: float,
    tau_loc: float,
    alpha0: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = sorted(cell_curve["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = map(int, groups)
    rows = []
    index_agg = aggregate_gaps.set_index(["split", "policy", "alpha"])

    for (split, policy, alpha, audit), sub in cell_curve.groupby(["split", "policy", "alpha", "audit"]):
        ga = sub[sub.group == a]
        gb = sub[sub.group == b]
        if ga.empty or gb.empty:
            continue
        ra = ga.iloc[0]
        rb = gb.iloc[0]
        agg = index_agg.loc[(split, policy, alpha)]
        delta = float(agg.delta_V_a_minus_b)
        delta_z = float(ra.avg_set_size - rb.avg_set_size)
        supported = bool(int(ra.n) >= n_min and int(rb.n) >= n_min)
        reversal = bool(supported and abs(delta) >= tau_agg and abs(delta_z) >= tau_loc and delta * delta_z < 0)
        burdened = a if delta_z >= 0 else b
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "audit": int(audit),
            "profession_id": int(ra.profession_id),
            "occupation_name": str(ra.occupation_name),
            "audit_name": str(ra.occupation_name),
            "group_a": a,
            "group_a_name": str(ra.group_name),
            "group_b": b,
            "group_b_name": str(rb.group_name),
            "n_a_z": int(ra.n),
            "n_b_z": int(rb.n),
            "V_a_z": float(ra.avg_set_size),
            "V_b_z": float(rb.avg_set_size),
            "C_a_z": float(ra.coverage),
            "C_b_z": float(rb.coverage),
            "aggregate_gap_delta": delta,
            "local_gap_delta_z": delta_z,
            "delta_times_delta_z": float(delta * delta_z),
            "reversal_strength": float(max(-delta * delta_z, 0.0)),
            "supported": supported,
            "is_simpson_reversal": reversal,
            "burdened_group": int(burdened),
            "burdened_group_name": str(ra.group_name if burdened == a else rb.group_name),
            "active_pair": f"({str(ra.group_name if burdened == a else rb.group_name)}, {str(ra.occupation_name)})" if reversal else "",
        })

    gap = pd.DataFrame(rows)
    if gap.empty:
        return gap, gap

    summaries = []
    for audit, sub in gap.groupby("audit"):
        sub = sub.sort_values("alpha")
        active = sub[sub.is_simpson_reversal].copy()
        nearest = sub.iloc[int(np.argmin(np.abs(sub.alpha.to_numpy(dtype=float) - float(alpha0))))]
        summaries.append({
            "audit": int(audit),
            "profession_id": int(sub.profession_id.iloc[0]),
            "occupation_name": str(sub.occupation_name.iloc[0]),
            "audit_name": str(sub.occupation_name.iloc[0]),
            "n_a_z": int(sub.n_a_z.iloc[0]),
            "n_b_z": int(sub.n_b_z.iloc[0]),
            "supported_all_alpha": bool(sub.supported.all()),
            "num_reversal_alpha": int(sub.is_simpson_reversal.sum()),
            "reversal_fraction": float(sub.is_simpson_reversal.mean()),
            "integrated_reversal_strength": float(sub.reversal_strength.mean()),
            "max_reversal_strength": float(sub.reversal_strength.max()),
            "max_abs_local_gap": float(np.max(np.abs(sub.local_gap_delta_z))),
            "min_delta_times_delta_z": float(sub.delta_times_delta_z.min()),
            "alpha0_nearest": float(nearest.alpha),
            "aggregate_gap_at_alpha0": float(nearest.aggregate_gap_delta),
            "local_gap_at_alpha0": float(nearest.local_gap_delta_z),
            "delta_times_delta_z_at_alpha0": float(nearest.delta_times_delta_z),
            "is_reversal_at_alpha0": bool(nearest.is_simpson_reversal),
            "alpha_values_with_reversal": ",".join(f"{x:g}" for x in active.alpha.tolist()),
        })

    summary = pd.DataFrame(summaries).sort_values(
        ["num_reversal_alpha", "integrated_reversal_strength", "max_abs_local_gap"],
        ascending=[False, False, False],
    )
    return gap, summary


def sample_level_table(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alpha: float,
    policy: str,
    meta: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    threshold = thresholds[policy][float(alpha)]
    size = set_sizes(split["S"], split["g"], threshold)
    cov = covered(split["S"], split["y"], split["g"], threshold)
    ts = true_scores(split["S"], split["y"])
    n = len(split["y"])

    def get_arr(key: str, default: Any) -> np.ndarray:
        if key in split and len(split[key]) == n:
            return np.asarray(split[key])
        if callable(default):
            return np.asarray([default(i) for i in range(n)])
        return np.asarray([default for _ in range(n)])

    idx = get_arr("idx", lambda i: i)
    artifact_index = get_arr("artifact_index", lambda i: idx[i])
    sample_id = get_arr("sample_id", lambda i: f"{split_name}:{i}")
    source_split = get_arr("source_split", "")
    source_index = get_arr("source_index", -1)

    rows = []
    for i in range(n):
        group = int(split["g"][i])
        audit = int(split["z"][i])
        label = int(split["y"][i])
        rows.append({
            "split": split_name,
            "policy": policy,
            "alpha": float(alpha),
            "row_in_split": int(i),
            "idx": str(idx[i]),
            "artifact_index": str(artifact_index[i]),
            "sample_id": str(sample_id[i]),
            "source_split": str(source_split[i]),
            "source_index": str(source_index[i]),
            "group": group,
            "group_name": group_name(meta, group),
            "audit": audit,
            "profession_id": audit_profession_id(meta, audit),
            "occupation_name": audit_name(meta, audit),
            "audit_name": audit_name(meta, audit),
            "y": label,
            "y_profession_id": audit_profession_id(meta, label),
            "y_name": audit_name(meta, label),
            "set_size": int(size[i]),
            "covered": bool(cov[i]),
            "true_score": float(ts[i]),
            "threshold": threshold_value(threshold, group),
        })
    return pd.DataFrame(rows)



def setup_plotting() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 360,
        "font.size": 13.0,
        "axes.titlesize": 13.2,
        "axes.labelsize": 12.2,
        "legend.fontsize": 10.2,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#F8FAFC",
        "figure.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#A8B3C7",
        "grid.linestyle": "-",
        "grid.linewidth": 1.05,
        "grid.alpha": 0.92,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def bright_palette() -> dict[str, str]:
    return {
        "male": "#0077FF",
        "female": "#FF4FA3",
        "aggregate": "#0F172A",
        "cell": "#F97316",
        "positive": "#10B981",
        "purple": "#7C3AED",
        "gold": "#F59E0B",
        "gray": "#64748B",
        "shade": "#FDE68A",
        "floor": "#8B5CF6",
        "geom": "#06B6D4",
        "slate": "#334155",
    }


def group_color(group_label: str, idx: int) -> str:
    p = bright_palette()
    key = str(group_label).lower()
    if "male" in key and "female" not in key:
        return p["male"]
    if "female" in key:
        return p["female"]
    return [p["male"], p["female"], p["purple"], p["gold"]][idx % 4]


def score_label(score: str) -> str:
    key = str(score).lower()
    if key == "saps":
        return "SAPS"
    if key == "raps":
        return "RAPS"
    if key == "simple":
        return "Simple"
    return str(score)


def local_marker_for_group(group_label: str, idx: int) -> str:
    key = str(group_label).lower()
    if "male" in key and "female" not in key:
        return "v"
    if "female" in key:
        return "^"
    return ["v", "^", "s", "D"][idx % 4]



def polish_axis(ax: Any, xgrid: bool = True, ygrid: bool = True) -> None:
    ax.set_facecolor("#F8FAFC")
    if xgrid and ygrid:
        axis = "both"
    elif xgrid:
        axis = "x"
    elif ygrid:
        axis = "y"
    else:
        axis = "both"
    ax.grid(True, axis=axis, alpha=0.92, color="#A8B3C7", linewidth=1.05)
    ax.tick_params(length=3.4, width=0.9, color="#475569")

def save_figure(fig: Any, stem: Path, formats: list[str]) -> list[str]:
    paths = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    return paths


def reversal_cmap() -> Any:
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        "bright_reversal",
        ["#F8FAFC", "#FEF3C7", "#FDBA74", "#FB7185", "#BE123C"],
    )



def add_heatmap_grid(ax: Any, n_rows: int, n_cols: int) -> None:
    ax.grid(False, which="major")
    ax.set_xticks(np.arange(-0.5, n_cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(True, which="minor", color="#CBD5E1", linewidth=0.95, alpha=0.98)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#CBD5E1")

def alpha_tick_positions(alpha_values: list[float], max_ticks: int = 10) -> tuple[list[int], list[str]]:
    n = len(alpha_values)
    if n <= max_ticks:
        idx = list(range(n))
    else:
        step = int(math.ceil(n / max_ticks))
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
    return idx, [f"{alpha_values[i]:g}" for i in idx]


def alpha_mid_widths(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size <= 1:
        return np.full_like(alpha, 0.01)
    diffs = np.diff(alpha)
    left = np.r_[diffs[0], diffs]
    right = np.r_[diffs, diffs[-1]]
    return 0.5 * (left + right)


def shade_reversal_regions(ax: Any, alpha: np.ndarray, active: np.ndarray, color: str | None = None, alpha_level: float = 0.22) -> None:
    p = bright_palette()
    color = color or p["shade"]
    alpha = np.asarray(alpha, dtype=float)
    active = np.asarray(active, dtype=bool)
    if alpha.size == 0 or not active.any():
        return
    widths = alpha_mid_widths(alpha)
    for a, w, is_active in zip(alpha, widths, active):
        if is_active:
            ax.axvspan(a - 0.5 * w, a + 0.5 * w, color=color, alpha=alpha_level, linewidth=0)


def audit_label_map_from_gap(gap: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    if gap.empty:
        return labels
    col = "occupation_name" if "occupation_name" in gap.columns else "audit_name"
    for row in gap[["audit", col]].drop_duplicates().itertuples(index=False):
        labels[int(row[0])] = clean_label(row[1])
    return labels


def audit_display_order(summary: pd.DataFrame, gap: pd.DataFrame, mode: str) -> list[int]:
    if mode == "label":
        return sorted(int(x) for x in gap.audit.unique())
    if summary is None or summary.empty:
        return sorted(int(x) for x in gap.audit.unique())
    d = summary.copy()
    cols = [c for c in ["num_reversal_alpha", "integrated_reversal_strength", "max_reversal_strength", "max_abs_local_gap"] if c in d.columns]
    if cols:
        d = d.sort_values(cols, ascending=False)
    return [int(x) for x in d.audit.tolist()]


def reversal_matrix(gap: pd.DataFrame, order: list[int], alpha_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    mat = np.zeros((len(order), len(alpha_values)), dtype=float)
    active = np.zeros((len(order), len(alpha_values)), dtype=bool)
    for i, audit in enumerate(order):
        sub = gap[gap.audit == audit]
        by_alpha = {float(r.alpha): r for r in sub.itertuples(index=False)}
        for j, alpha in enumerate(alpha_values):
            row = by_alpha.get(float(alpha))
            if row is None:
                continue
            if bool(row.is_simpson_reversal):
                mat[i, j] = abs(float(row.local_gap_delta_z))
                active[i, j] = True
    return mat, active


def strongest_witness(gap: pd.DataFrame) -> pd.Series:
    active = gap[gap.is_simpson_reversal].copy()
    if not active.empty:
        return active.iloc[int(active.reversal_strength.argmax())]
    return gap.iloc[int(gap.reversal_strength.argmax())]



def plot_main_figure(
    curve: pd.DataFrame,
    gap: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    score: str,
    formats: list[str],
    order_mode: str,
) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if gap.empty:
        return []
    p = bright_palette()
    score_name = score_label(score)
    alpha_values = sorted(float(x) for x in gap.alpha.unique())
    order = audit_display_order(summary, gap, order_mode)
    labels = audit_label_map_from_gap(gap)
    y_labels = [clean_label(labels.get(a, str(a))) for a in order]
    mat, active_mat = reversal_matrix(gap, order, alpha_values)

    fig = plt.figure(figsize=(18.2, 9.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.03, 1.34], height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    dplot = curve[(curve.split == "repair") & (curve.policy == "groupwise")].copy()
    for idx, (gname, sub) in enumerate(dplot.groupby("group_name", sort=False)):
        sub = sub.sort_values("alpha")
        ax_a.plot(
            sub.alpha,
            sub.avg_set_size,
            marker="o",
            markersize=5.2,
            linewidth=3.0,
            color=group_color(str(gname), idx),
            label=str(gname),
        )
    ax_a.set_title("A. Aggregate set-size curve on D_repair", loc="left", fontweight="bold")
    ax_a.set_xlabel("miscoverage level")
    ax_a.set_ylabel("average set size")
    polish_axis(ax_a)
    ax_a.legend(title="group", loc="best", frameon=True, framealpha=0.94)

    vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.0
    cmap = reversal_cmap()
    im = ax_b.imshow(mat, aspect="auto", interpolation="none", vmin=0.0, vmax=max(vmax, 1e-12), cmap=cmap)
    xticks, xticklabels = alpha_tick_positions(alpha_values, max_ticks=12)
    ax_b.set_xticks(xticks)
    ax_b.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(y_labels)
    ax_b.set_xlabel("miscoverage level")
    ax_b.set_ylabel("occupation")
    ax_b.set_title("B. Simpson reversal evidence", loc="left", fontweight="bold")
    add_heatmap_grid(ax_b, len(order), len(alpha_values))
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.034, pad=0.018)
    cbar.set_label("reversal strength = abs(local gap)")

    if summary is not None and not summary.empty:
        counts = summary.set_index("audit").reindex(order).reset_index()
        vals = counts["num_reversal_alpha"].fillna(0).astype(float).to_numpy()
        strength = counts["max_abs_local_gap"].fillna(0.0).astype(float).to_numpy()
    else:
        vals = active_mat.sum(axis=1).astype(float)
        strength = mat.max(axis=1)
    y = np.arange(len(order))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(float(np.nanmax(strength)), 1e-12))
    colors = [reversal_cmap()(norm(st)) if v > 0 else "#CBD5E1" for v, st in zip(vals, strength)]
    ax_c.barh(y, vals, color=colors, height=0.70, edgecolor="white", linewidth=0.9)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(y_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel(f"number of grid levels with Simpson reversal (out of {len(alpha_values)})")
    ax_c.set_title("C. Reversal persistence by occupation", loc="left", fontweight="bold")
    polish_axis(ax_c, xgrid=True, ygrid=False)
    max_val = max(float(np.nanmax(vals)), 1.0)
    for yi, val in zip(y, vals):
        ax_c.text(val + 0.16, yi, f"{int(val)} of {len(alpha_values)}", va="center", fontsize=10.0, color="#334155")
    ax_c.set_xlim(0, max_val + 3.2)

    chosen = strongest_witness(gap)
    audit = int(chosen.audit)
    label = clean_label(labels.get(audit, str(audit)))
    local = gap[gap.audit == audit].sort_values("alpha").copy()
    agg_curve = gap[["alpha", "aggregate_gap_delta"]].drop_duplicates().sort_values("alpha").copy()
    alpha = local.alpha.to_numpy(dtype=float)
    local_y = local.local_gap_delta_z.to_numpy(dtype=float)
    agg_y = agg_curve.set_index("alpha").reindex(local.alpha)["aggregate_gap_delta"].to_numpy(dtype=float)
    all_y = np.r_[local_y[np.isfinite(local_y)], agg_y[np.isfinite(agg_y)], [0.0]]
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = max(0.025, 0.12 * (ymax - ymin + 1e-12))
    shade_reversal_regions(ax_d, alpha, local.is_simpson_reversal.to_numpy(dtype=bool), color=p["shade"], alpha_level=0.27)
    gap_label = f"{str(chosen.group_a_name)} - {str(chosen.group_b_name)}"
    ax_d.plot(alpha, agg_y, linestyle="-", marker="o", markersize=4.5, linewidth=2.8, color=p["aggregate"], label=f"aggregate gap ({gap_label})")
    ax_d.plot(alpha, local_y, linestyle="-", marker="o", markersize=4.5, linewidth=2.8, color=p["cell"], label=f"{label} gap ({gap_label})")
    ax_d.axhline(0.0, linewidth=1.25, color="#0F172A")
    ax_d.set_ylim(ymin - pad, ymax + pad)
    ax_d.set_xlabel("miscoverage level")
    ax_d.set_ylabel(f"set-size gap = {gap_label}")
    ax_d.set_title(f"D. Strongest witness: {label}", loc="left", fontweight="bold")
    polish_axis(ax_d)
    ax_d.legend(loc="best", frameon=True, framealpha=0.94)

    fig.suptitle(f"BioBias curve diagnosis ({score_name})", y=0.992, fontsize=17.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, out_dir / f"fig_main_biobias_curve_simpson_{score}", formats)


def plot_occupation_curve_grid(
    cell: pd.DataFrame,
    curve: pd.DataFrame,
    gap: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    score: str,
    formats: list[str],
    order_mode: str,
) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    if cell.empty or gap.empty:
        return []
    p = bright_palette()
    score_name = score_label(score)
    order = audit_display_order(summary, gap, order_mode)
    labels = audit_label_map_from_gap(gap)
    local = cell[(cell.split == "repair") & (cell.policy == "groupwise")].copy()
    agg = curve[(curve.split == "repair") & (curve.policy == "groupwise")].copy().sort_values(["group", "alpha"])

    y_values = local.avg_set_size.astype(float).tolist() + agg.avg_set_size.astype(float).tolist()
    finite_y = np.asarray([v for v in y_values if np.isfinite(v)], dtype=float)
    if finite_y.size == 0:
        return []
    lo, hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
    pad = max(0.10, 0.10 * (hi - lo + 1e-12))
    ylim = (max(0.0, lo - pad), hi + pad)

    fig, axes = plt.subplots(2, 5, figsize=(18.2, 8.25), sharex=True, sharey=True)
    axes = axes.ravel()
    group_order = sorted(local.group.unique())
    summary_counts = summary.set_index("audit")["num_reversal_alpha"].to_dict() if summary is not None and not summary.empty else {}

    for ax, audit in zip(axes, order):
        sub_gap = gap[gap.audit == audit].sort_values("alpha")
        alpha = sub_gap.alpha.to_numpy(dtype=float)
        shade_reversal_regions(ax, alpha, sub_gap.is_simpson_reversal.to_numpy(dtype=bool), color=p["shade"], alpha_level=0.22)
        for idx, group in enumerate(group_order):
            gname = str(local[local.group == group].group_name.iloc[0])
            color = group_color(gname, idx)
            ref = agg[agg.group == group].sort_values("alpha")
            ax.plot(
                ref.alpha,
                ref.avg_set_size,
                linestyle="-",
                marker="o",
                markersize=3.9,
                markerfacecolor=color,
                markeredgecolor=color,
                linewidth=2.25,
                color=color,
                alpha=0.90,
                label=f"{gname} aggregate" if audit == order[0] else None,
            )
            sub = local[(local.audit == audit) & (local.group == group)].sort_values("alpha")
            ax.plot(
                sub.alpha,
                sub.avg_set_size,
                linestyle="--",
                marker=local_marker_for_group(gname, idx),
                markersize=4.6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.65,
                linewidth=1.75,
                color=color,
                alpha=0.98,
                label=f"{gname} occupation" if audit == order[0] else None,
            )
        n_rows = local[local.audit == audit].drop_duplicates(["group"])
        n_text = ", ".join(f"{str(r.group_name)[0]}={int(r.n)}" for r in n_rows.itertuples(index=False))
        reversals = int(summary_counts.get(audit, int(sub_gap.is_simpson_reversal.sum())))
        ax.set_title(f"{clean_label(labels.get(audit, audit))}\n{reversals}/{len(sub_gap)} Simpson reversals; {n_text}", fontsize=10.0, fontweight="bold")
        ax.set_ylim(*ylim)
        polish_axis(ax)

    for ax in axes[len(order):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(order)]):
        if i % 5 == 0:
            ax.set_ylabel("average set size")
        else:
            ax.set_ylabel("")
        if i >= 5:
            ax.set_xlabel("miscoverage level")
        else:
            ax.set_xlabel("")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"Occupation-conditioned curves with aggregate references ({score_name})", y=1.065, fontsize=16.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, out_dir / f"fig_appendix_occupation_curve_grid_{score}", formats)


def plot_gap_curve_grid(
    gap: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    score: str,
    formats: list[str],
    order_mode: str,
) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    if gap.empty:
        return []
    p = bright_palette()
    score_name = score_label(score)
    order = audit_display_order(summary, gap, order_mode)
    labels = audit_label_map_from_gap(gap)
    all_gap_values = gap[["aggregate_gap_delta", "local_gap_delta_z"]].to_numpy(dtype=float).ravel()
    finite = all_gap_values[np.isfinite(all_gap_values)]
    if finite.size == 0:
        return []
    bound = max(0.05, float(np.nanmax(np.abs(finite))) * 1.12)

    fig, axes = plt.subplots(2, 5, figsize=(18.2, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, audit in zip(axes, order):
        sub = gap[gap.audit == audit].sort_values("alpha")
        alpha = sub.alpha.to_numpy(dtype=float)
        shade_reversal_regions(ax, alpha, sub.is_simpson_reversal.to_numpy(dtype=bool), color=p["shade"], alpha_level=0.22)
        ax.axhline(0.0, color="#0F172A", linewidth=1.1)
        ax.plot(alpha, sub.aggregate_gap_delta.to_numpy(dtype=float), color=p["aggregate"], linewidth=2.15, marker="o", markersize=3.6, label="aggregate gap" if audit == order[0] else None)
        ax.plot(alpha, sub.local_gap_delta_z.to_numpy(dtype=float), color=p["cell"], linewidth=1.9, linestyle="--", marker="D", markersize=3.3, label="occupation gap" if audit == order[0] else None)
        reversals = int(sub.is_simpson_reversal.sum())
        ax.set_title(f"{clean_label(labels.get(audit, audit))}\n{reversals}/{len(sub)} Simpson reversals", fontsize=10.0, fontweight="bold")
        ax.set_ylim(-bound, bound)
        polish_axis(ax)

    for ax in axes[len(order):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(order)]):
        if i % 5 == 0:
            ax.set_ylabel("set-size gap")
        else:
            ax.set_ylabel("")
        if i >= 5:
            ax.set_xlabel("miscoverage level")
        else:
            ax.set_xlabel("")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"Appendix: aggregate gap versus occupation gap curves ({score_name})", y=1.065, fontsize=16.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, out_dir / f"fig_appendix_gap_curve_grid_{score}", formats)

def plot_support_counts_global(raw_repair: dict[str, np.ndarray], meta: dict[str, Any], out_root: Path, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    z = raw_repair["z"]
    g = raw_repair["g"]
    audits = sorted(int(x) for x in np.unique(z))
    groups = sorted(int(x) for x in np.unique(g))
    mat = np.zeros((len(audits), len(groups)), dtype=float)
    for i, audit in enumerate(audits):
        for j, group in enumerate(groups):
            mat[i, j] = float(np.sum((z == audit) & (g == group)))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap="YlGnBu")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([group_name(meta, group) for group in groups])
    ax.set_yticks(range(len(audits)))
    ax.set_yticklabels([clean_label(audit_name(meta, audit)) for audit in audits])
    ax.set_xlabel("group")
    ax.set_ylabel("occupation")
    ax.set_title("Appendix: D_repair support counts", loc="left", fontweight="bold")
    add_heatmap_grid(ax, len(audits), len(groups))
    for i in range(len(audits)):
        for j in range(len(groups)):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=8.5, color="#0F172A")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("sample count")
    fig.tight_layout()
    figures_dir = mkdir(out_root / "figures")
    return save_figure(fig, figures_dir / "fig_appendix_support_counts", formats)



def plot_score_robustness_heatmaps(out_root: Path, scores: list[str], formats: list[str], order_mode: str) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    gaps: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    for score in scores:
        gap_path = out_root / score / "tables" / "simpson_gap_curve_long.csv"
        summary_path = out_root / score / "tables" / "simpson_curve_summary.csv"
        if gap_path.exists() and summary_path.exists():
            gaps[score] = pd.read_csv(gap_path)
            summaries[score] = pd.read_csv(summary_path)
    if len(gaps) <= 1:
        return []

    base_score = scores[0]
    base_gap = gaps[base_score]
    base_summary = summaries[base_score]
    order = audit_display_order(base_summary, base_gap, order_mode)
    labels = audit_label_map_from_gap(base_gap)
    alpha_values = sorted(float(x) for x in base_gap.alpha.unique())

    mats = []
    vmax = 0.0
    for score in scores:
        mat, _ = reversal_matrix(gaps[score], order, alpha_values)
        mats.append(mat)
        vmax = max(vmax, float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.0)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(1, len(scores), figsize=(5.7 * len(scores), 5.85), sharey=True)
    if len(scores) == 1:
        axes = np.asarray([axes])
    for ax, score, mat in zip(axes, scores, mats):
        im = ax.imshow(mat, aspect="auto", interpolation="none", vmin=0.0, vmax=vmax, cmap=reversal_cmap())
        xticks, xticklabels = alpha_tick_positions(alpha_values, max_ticks=8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right")
        ax.set_title(score_label(score), fontweight="bold")
        ax.set_xlabel("miscoverage level")
        add_heatmap_grid(ax, len(order), len(alpha_values))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([clean_label(labels.get(a, str(a))) for a in order])
    axes[0].set_ylabel("occupation")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("reversal strength")
    fig.suptitle("Appendix: score robustness of Simpson curve diagnosis", y=1.02, fontsize=16.0, fontweight="bold")
    figures_dir = mkdir(out_root / "figures")
    return save_figure(fig, figures_dir / "fig_appendix_score_robustness_heatmaps", formats)

def binary_entropy(alpha: float, eps: float = 1e-12) -> float:
    a = min(max(float(alpha), eps), 1.0 - eps)
    return float(-a * math.log(a) - (1.0 - a) * math.log(1.0 - a))


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def entropy_floor_value(h_cond: float, alpha: float, k: int) -> float:
    if k <= 1:
        return 1.0
    rhs = float(h_cond) - binary_entropy(float(alpha)) - float(alpha) * math.log(float(k - 1))
    return float(max(math.exp(rhs), 0.0))



def entropy_floor_table(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    split_name: str,
    policy: str,
) -> pd.DataFrame:
    rows = []
    probs = np.asarray(split["probs"], dtype=float)
    ent = predictive_entropy(probs)
    k = int(probs.shape[1])
    scopes: list[tuple[str, int | None, str, np.ndarray]] = [("all", None, "All", np.ones(len(ent), dtype=bool))]
    for group in sorted(np.unique(split["g"])):
        scopes.append(("group", int(group), group_name(meta, int(group)), split["g"] == group))

    for alpha in alphas:
        threshold = thresholds[policy][alpha]
        sizes = set_sizes(split["S"], split["g"], threshold).astype(float)
        psi = binary_entropy(alpha) + float(alpha) * math.log(float(max(k - 1, 1))) if k > 1 else binary_entropy(alpha)
        error_channel_volume = float(math.exp(psi))
        for scope, group, label, mask in scopes:
            if not np.any(mask):
                continue
            scoped_sizes = sizes[mask]
            scoped_sizes_plus = np.maximum(scoped_sizes, 1.0)
            h_cond = float(np.mean(ent[mask]))
            intrinsic_volume = float(math.exp(h_cond))
            floor = entropy_floor_value(h_cond, alpha, k)
            mean_size = float(np.mean(scoped_sizes))
            mean_size_plus = float(np.mean(scoped_sizes_plus))
            empty_rate = float(np.mean(scoped_sizes <= 0.0))
            if empty_rate > 0.0:
                geom_size = 0.0
                log_geom = np.nan
                log_product_slack = np.nan
                product_volume = 0.0
            else:
                log_geom = float(np.mean(np.log(scoped_sizes)))
                geom_size = float(math.exp(log_geom))
                log_product_slack = float(log_geom + psi - h_cond)
                product_volume = float(geom_size * error_channel_volume)
            log_geom_plus = float(np.mean(np.log(scoped_sizes_plus)))
            geom_size_plus = float(math.exp(log_geom_plus))
            log_product_slack_plus = float(log_geom_plus + psi - h_cond)
            product_volume_plus = float(geom_size_plus * error_channel_volume)
            floor_safe = max(float(floor), 1e-12)
            log_floor = float(math.log(floor_safe))
            rows.append({
                "split": split_name,
                "policy": policy,
                "alpha": float(alpha),
                "scope": scope,
                "group": int(group) if group is not None else -1,
                "group_name": label,
                "n": int(mask.sum()),
                "num_labels": int(k),
                "plugin_conditional_entropy": h_cond,
                "intrinsic_log_volume": h_cond,
                "intrinsic_volume": intrinsic_volume,
                "binary_entropy_alpha": binary_entropy(alpha),
                "error_channel_log_volume": float(psi),
                "error_channel_volume": error_channel_volume,
                "entropy_floor": float(floor),
                "arithmetic_mean_set_size": mean_size,
                "geometric_mean_set_size": geom_size,
                "log_geometric_mean_set_size": log_geom,
                "completed_arithmetic_mean_set_size": mean_size_plus,
                "completed_geometric_mean_set_size": geom_size_plus,
                "completed_log_geometric_mean_set_size": log_geom_plus,
                "empty_rate": empty_rate,
                "product_volume": product_volume,
                "excess_log_product_volume": log_product_slack,
                "completed_product_volume": product_volume_plus,
                "completed_excess_log_product_volume": log_product_slack_plus,
                "log_slack_geometric_over_floor": float(log_geom - log_floor) if np.isfinite(log_geom) else np.nan,
                "completed_log_slack_geometric_over_floor": float(log_geom_plus - log_floor),
                "ratio_geometric_over_floor": float(geom_size / floor_safe) if geom_size > 0.0 else 0.0,
                "completed_ratio_geometric_over_floor": float(geom_size_plus / floor_safe),
                "ratio_arithmetic_over_floor": float(mean_size / floor_safe),
                "completed_ratio_arithmetic_over_floor": float(mean_size_plus / floor_safe),
            })
    return pd.DataFrame(rows)


def _finite_product_df(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out[x_col] = out[x_col].astype(float)
    out[y_col] = out[y_col].astype(float)
    return out[np.isfinite(out[x_col]) & np.isfinite(out[y_col])].copy()


def _minimum_slack_row(df: pd.DataFrame) -> pd.Series | None:
    finite = _finite_product_df(df, "alpha", "product_slack")
    if finite.empty:
        return None
    idx = finite["product_slack"].astype(float).idxmin()
    return finite.loc[idx]


def _scatter_minimum_point(ax: Any, x: float, y: float, color: str, label: str | None = None, size: float = 110.0) -> None:
    ax.scatter(
        [float(x)],
        [float(y)],
        s=size,
        marker="*",
        color=color,
        edgecolor="white",
        linewidth=1.05,
        zorder=6,
        label=label,
    )


def _plot_finite_line(ax: Any, x: np.ndarray, y: np.ndarray, *, color: str, label: str, linewidth: float, marker: str = "o", markersize: float = 4.7, alpha: float = 1.0, zorder: int = 3) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    ax.plot(x[mask], y[mask], color=color, linewidth=linewidth, marker=marker, markersize=markersize, alpha=alpha, label=label, zorder=zorder)


def plot_entropy_floor_figure(entropy_df: pd.DataFrame, out_dir: Path, score: str, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if entropy_df.empty:
        return []
    p = bright_palette()
    score_name = score_label(score)
    df = entropy_df[(entropy_df.split == "repair") & (entropy_df.policy == "groupwise")].copy()
    if df.empty:
        df = entropy_df.copy()
    eps = 1e-12
    df["log_V"] = np.log(np.maximum(df["arithmetic_mean_set_size"].astype(float), eps))
    if "log_geometric_mean_set_size" not in df.columns:
        df["log_geometric_mean_set_size"] = np.where(df["geometric_mean_set_size"].astype(float) > 0.0, np.log(df["geometric_mean_set_size"].astype(float)), np.nan)
    df["log_G"] = df["log_geometric_mean_set_size"].astype(float)
    df["log_floor"] = np.log(np.maximum(df["entropy_floor"].astype(float), eps))
    df["product_slack"] = df.get("excess_log_product_volume", df["log_G"] - df["log_floor"])

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.45))
    ax_a, ax_b = axes

    all_df = df[df.scope == "all"].sort_values("alpha").copy()
    min_row = _minimum_slack_row(all_df)
    if not all_df.empty:
        hhat = float(all_df["plugin_conditional_entropy"].iloc[0])
        finite = _finite_product_df(all_df, "error_channel_log_volume", "log_G")
        psi_all = all_df["error_channel_log_volume"].astype(float).to_numpy()
        if not finite.empty:
            x = finite["error_channel_log_volume"].to_numpy(dtype=float)
            y = finite["log_G"].to_numpy(dtype=float)
            a = finite["alpha"].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y, a = x[order], y[order], a[order]
            xmin = float(np.nanmin(x))
            xmax = float(np.nanmax(x))
            pad_x = max(0.012, 0.06 * (xmax - xmin + 1e-9))
            xx = np.linspace(xmin - 0.15 * pad_x, xmax + 0.15 * pad_x, 220)
            boundary = hhat - xx
            ax_a.plot(xx, boundary, color=p["floor"], linewidth=2.7, linestyle="--", label=r"boundary: $\log G=\widehat H-\Psi_K$")
            ax_a.fill_between(xx, min(float(np.nanmin(boundary)) - 0.05, -0.35), boundary, color=p["floor"], alpha=0.09, linewidth=0)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1) if len(points) >= 2 else np.empty((0, 2, 2))
            norm = plt.Normalize(float(np.nanmin(a)), float(np.nanmax(a))) if len(a) > 1 else plt.Normalize(float(a[0]) - 1e-3, float(a[0]) + 1e-3)
            if len(segments) > 0:
                lc = LineCollection(segments, cmap="viridis", norm=norm)
                lc.set_array(a[:-1])
                lc.set_linewidth(3.0)
                ax_a.add_collection(lc)
            sc = ax_a.scatter(x, y, c=a, cmap="viridis", norm=norm, s=48, edgecolor="white", linewidth=0.75, zorder=4)
            cbar = fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.018)
            cbar.set_label("miscoverage level", fontsize=11.4)
            if min_row is not None and np.isfinite(float(min_row.get("log_G", np.nan))):
                _scatter_minimum_point(
                    ax_a,
                    float(min_row["error_channel_log_volume"]),
                    float(min_row["log_G"]),
                    p["aggregate"],
                    label="minimum slack",
                    size=130.0,
                )
            y_candidates = [float(np.nanmin(y)), float(np.nanmax(y)), float(np.nanmin(boundary)), float(np.nanmax(boundary))]
            yr0, yr1 = min(y_candidates), max(y_candidates)
            pad_y = max(0.06, 0.09 * (yr1 - yr0 + 1e-9))
            ax_a.set_ylim(yr0 - pad_y, yr1 + pad_y)
            ax_a.set_xlim(xmin - pad_x, xmax + pad_x)
        else:
            ax_a.text(0.5, 0.5, "Raw G is undefined\nwhen empty sets occur", ha="center", va="center", transform=ax_a.transAxes, fontsize=12.2, fontweight="bold")
    ax_a.set_title(r"A. All-sample product curve: $\log G+\Psi_K \geq \widehat H$", loc="left", fontweight="bold")
    ax_a.set_xlabel(r"error-channel log-volume $\Psi_K$")
    ax_a.set_ylabel(r"prediction log-volume $\log G$")
    polish_axis(ax_a)
    handles, labels = ax_a.get_legend_handles_labels()
    if handles:
        ax_a.legend(handles, labels, loc="best", fontsize=10.0, frameon=True, framealpha=0.94)

    all_line = df[df.scope == "all"].sort_values("alpha").copy()
    if not all_line.empty:
        alpha_all = all_line.alpha.to_numpy(dtype=float)
        slack_all = all_line["product_slack"].to_numpy(dtype=float)
        _plot_finite_line(ax_b, alpha_all, slack_all, color=p["aggregate"], linewidth=3.05, marker="o", markersize=5.2, label="All")
        invalid = ~np.isfinite(slack_all)
        if np.any(invalid):
            invalid_start = float(np.nanmin(alpha_all[invalid]))
            invalid_end = float(np.nanmax(alpha_all))
            ax_b.axvspan(invalid_start, invalid_end, color="#CBD5E1", alpha=0.24, linewidth=0)
            ax_b.text(
                invalid_start + 0.002,
                0.96,
                "raw G invalid",
                ha="left",
                va="top",
                transform=ax_b.get_xaxis_transform(),
                fontsize=10.4,
                color="#334155",
                fontweight="bold",
            )
        if min_row is not None and np.isfinite(float(min_row.get("product_slack", np.nan))):
            _scatter_minimum_point(ax_b, float(min_row["alpha"]), float(min_row["product_slack"]), p["aggregate"], label="minimum slack", size=130.0)
            ax_b.axvline(float(min_row["alpha"]), color=p["aggregate"], linewidth=1.25, linestyle=":", alpha=0.75)
    ax_b.axhline(0.0, color="#0F172A", linewidth=1.25)
    ax_b.set_title("B. Excess all-sample log-product volume", loc="left", fontweight="bold")
    ax_b.set_xlabel("miscoverage level")
    ax_b.set_ylabel(r"$\log G+\Psi_K-\widehat H$")
    polish_axis(ax_b)
    finite_slack = all_line[np.isfinite(all_line["product_slack"].astype(float))].copy()
    if not finite_slack.empty:
        y0 = float(finite_slack["product_slack"].min())
        y1 = float(finite_slack["product_slack"].max())
        pad = max(0.035, 0.12 * (y1 - y0 + 1e-9))
        ax_b.set_ylim(min(-0.01, y0 - pad), y1 + pad)
    alpha_vals = sorted(float(x) for x in df["alpha"].dropna().unique())
    if alpha_vals:
        ax_b.set_xlim(min(alpha_vals) - 0.003, max(alpha_vals) + 0.003)
    ax_b.legend(loc="best", fontsize=10.0, frameon=True, framealpha=0.94)

    fig.suptitle(f"Product-type uncertainty diagnostic for BioBias ({score_name})", y=1.02, fontsize=17.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    paths = save_figure(fig, out_dir / f"fig_product_uncertainty_{score}", formats)
    return paths



def plot_product_uncertainty_gplus_figure(entropy_df: pd.DataFrame, out_dir: Path, score: str, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if entropy_df.empty:
        return []
    p = bright_palette()
    score_name = score_label(score)
    df = entropy_df[(entropy_df.split == "repair") & (entropy_df.policy == "groupwise")].copy()
    if df.empty:
        df = entropy_df.copy()
    required = {"completed_log_geometric_mean_set_size", "completed_excess_log_product_volume"}
    if not required.issubset(set(df.columns)):
        return []

    df["log_G_plus"] = df["completed_log_geometric_mean_set_size"].astype(float)
    df["product_slack_plus"] = df["completed_excess_log_product_volume"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.45))
    ax_a, ax_b = axes

    all_df = df[df.scope == "all"].sort_values("alpha").copy()
    min_row = None
    if not all_df.empty:
        finite_for_min = _finite_product_df(all_df.rename(columns={"product_slack_plus": "product_slack"}), "alpha", "product_slack")
        if not finite_for_min.empty:
            min_idx = finite_for_min["product_slack"].astype(float).idxmin()
            min_row = all_df.loc[min_idx]

        hhat = float(all_df["plugin_conditional_entropy"].iloc[0])
        finite = _finite_product_df(all_df, "error_channel_log_volume", "log_G_plus")
        if not finite.empty:
            x = finite["error_channel_log_volume"].to_numpy(dtype=float)
            y = finite["log_G_plus"].to_numpy(dtype=float)
            a = finite["alpha"].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y, a = x[order], y[order], a[order]
            xmin = float(np.nanmin(x))
            xmax = float(np.nanmax(x))
            pad_x = max(0.012, 0.06 * (xmax - xmin + 1e-9))
            xx = np.linspace(xmin - 0.15 * pad_x, xmax + 0.15 * pad_x, 220)
            boundary = hhat - xx
            ax_a.plot(xx, boundary, color=p["floor"], linewidth=2.7, linestyle="--", label=r"boundary: $\log G_+=\widehat H-\Psi_K$")
            ax_a.fill_between(xx, min(float(np.nanmin(boundary)) - 0.05, -0.35), boundary, color=p["floor"], alpha=0.09, linewidth=0)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1) if len(points) >= 2 else np.empty((0, 2, 2))
            norm = plt.Normalize(float(np.nanmin(a)), float(np.nanmax(a))) if len(a) > 1 else plt.Normalize(float(a[0]) - 1e-3, float(a[0]) + 1e-3)
            if len(segments) > 0:
                lc = LineCollection(segments, cmap="viridis", norm=norm)
                lc.set_array(a[:-1])
                lc.set_linewidth(3.0)
                ax_a.add_collection(lc)
            sc = ax_a.scatter(x, y, c=a, cmap="viridis", norm=norm, s=48, edgecolor="white", linewidth=0.75, zorder=4)
            cbar = fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.018)
            cbar.set_label("miscoverage level", fontsize=11.4)
            if min_row is not None and np.isfinite(float(min_row.get("log_G_plus", np.nan))):
                _scatter_minimum_point(
                    ax_a,
                    float(min_row["error_channel_log_volume"]),
                    float(min_row["log_G_plus"]),
                    p["aggregate"],
                    label="minimum slack",
                    size=130.0,
                )
            y_candidates = [float(np.nanmin(y)), float(np.nanmax(y)), float(np.nanmin(boundary)), float(np.nanmax(boundary))]
            yr0, yr1 = min(y_candidates), max(y_candidates)
            pad_y = max(0.06, 0.09 * (yr1 - yr0 + 1e-9))
            ax_a.set_ylim(yr0 - pad_y, yr1 + pad_y)
            ax_a.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_a.set_title(r"A. Completed-set product curve: $\log G_+ + \Psi_K \geq \widehat H$", loc="left", fontweight="bold")
    ax_a.set_xlabel(r"error-channel log-volume $\Psi_K$")
    ax_a.set_ylabel(r"prediction log-volume $\log G_+$")
    polish_axis(ax_a)
    handles, labels = ax_a.get_legend_handles_labels()
    if handles:
        ax_a.legend(handles, labels, loc="best", fontsize=10.0, frameon=True, framealpha=0.94)

    all_line = df[df.scope == "all"].sort_values("alpha").copy()
    if not all_line.empty:
        alpha_all = all_line.alpha.to_numpy(dtype=float)
        slack_all = all_line["product_slack_plus"].to_numpy(dtype=float)
        _plot_finite_line(ax_b, alpha_all, slack_all, color=p["aggregate"], linewidth=3.05, marker="o", markersize=5.2, label="All")
        if min_row is not None and np.isfinite(float(min_row.get("product_slack_plus", np.nan))):
            _scatter_minimum_point(ax_b, float(min_row["alpha"]), float(min_row["product_slack_plus"]), p["aggregate"], label="minimum slack", size=130.0)
            ax_b.axvline(float(min_row["alpha"]), color=p["aggregate"], linewidth=1.25, linestyle=":", alpha=0.75)
        for idx, (gname, sub) in enumerate(df[df.scope == "group"].groupby("group_name", sort=False)):
            sub = sub.sort_values("alpha")
            _plot_finite_line(
                ax_b,
                sub.alpha.to_numpy(dtype=float),
                sub["product_slack_plus"].to_numpy(dtype=float),
                color=group_color(str(gname), idx),
                linewidth=2.45,
                marker="o",
                markersize=4.5,
                label=str(gname),
                alpha=0.95,
            )
    ax_b.axhline(0.0, color="#0F172A", linewidth=1.25)
    ax_b.set_title(r"B. Excess completed log-product volume", loc="left", fontweight="bold")
    ax_b.set_xlabel("miscoverage level")
    ax_b.set_ylabel(r"$\log G_+ + \Psi_K - \widehat H$")
    polish_axis(ax_b)
    finite_slack = all_line[np.isfinite(all_line["product_slack_plus"].astype(float))].copy() if not all_line.empty else pd.DataFrame()
    if not finite_slack.empty:
        y0 = float(finite_slack["product_slack_plus"].min())
        y1 = float(finite_slack["product_slack_plus"].max())
        pad = max(0.035, 0.12 * (y1 - y0 + 1e-9))
        ax_b.set_ylim(min(-0.01, y0 - pad), y1 + pad)
    alpha_vals = sorted(float(x) for x in df["alpha"].dropna().unique())
    if alpha_vals:
        ax_b.set_xlim(min(alpha_vals) - 0.003, max(alpha_vals) + 0.003)
    ax_b.legend(loc="best", fontsize=10.0, frameon=True, framealpha=0.94)

    fig.suptitle(f"Product-type uncertainty diagnostic for BioBias ({score_name}, completed sets)", y=1.02, fontsize=17.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    paths = save_figure(fig, out_dir / f"fig_product_uncertainty_gplus_{score}", formats)
    return paths

def plot_empty_set_rate_figure(entropy_df: pd.DataFrame, out_dir: Path, score: str, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    if entropy_df.empty:
        return []
    df = entropy_df[(entropy_df.split == "repair") & (entropy_df.policy == "groupwise")].copy()
    if df.empty:
        df = entropy_df.copy()
    if df.empty or "empty_rate" not in df.columns:
        return []
    rates = df["empty_rate"].astype(float).to_numpy()
    if not np.any(np.isfinite(rates)) or float(np.nanmax(rates)) <= 0.0:
        return []
    p = bright_palette()
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.7))
    all_df = df[df.scope == "all"].sort_values("alpha")
    if not all_df.empty:
        ax.plot(all_df.alpha, all_df.empty_rate, color=p["aggregate"], linewidth=2.8, marker="o", markersize=5.0, label="All")
    for idx, (gname, sub) in enumerate(df[df.scope == "group"].groupby("group_name", sort=False)):
        sub = sub.sort_values("alpha")
        ax.plot(sub.alpha, sub.empty_rate, color=group_color(str(gname), idx), linewidth=2.35, marker="o", markersize=4.4, label=str(gname))
    ax.set_title(f"Appendix: raw empty-set rate for {score_label(score)}", loc="left", fontweight="bold")
    ax.set_xlabel("miscoverage level")
    ax.set_ylabel("fraction of empty sets")
    polish_axis(ax)
    ax.legend(loc="best", fontsize=10.2, frameon=True, framealpha=0.94)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_appendix_empty_set_rate_{score}", formats)



def cell_product_slack_table(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    split_name: str,
    policy: str,
) -> pd.DataFrame:
    rows = []
    probs = np.asarray(split["probs"], dtype=float)
    ent = predictive_entropy(probs)
    k = int(probs.shape[1])
    z_arr = np.asarray(split["z"], dtype=int)
    g_arr = np.asarray(split["g"], dtype=int)
    audits = sorted(int(x) for x in np.unique(z_arr))
    groups = sorted(int(x) for x in np.unique(g_arr))
    for alpha in alphas:
        threshold = thresholds[policy][float(alpha)]
        sizes = set_sizes(split["S"], split["g"], threshold).astype(float)
        psi = binary_entropy(alpha) + float(alpha) * math.log(float(max(k - 1, 1))) if k > 1 else binary_entropy(alpha)
        for audit in audits:
            for group in groups:
                mask = (z_arr == audit) & (g_arr == group)
                if not np.any(mask):
                    continue
                cell_sizes = sizes[mask]
                cell_sizes_plus = np.maximum(cell_sizes, 1.0)
                empty_rate = float(np.mean(cell_sizes <= 0.0))
                h_cell = float(np.mean(ent[mask]))
                log_g_plus = float(np.mean(np.log(cell_sizes_plus)))
                g_plus = float(math.exp(log_g_plus))
                slack_plus = float(log_g_plus + psi - h_cell)
                if empty_rate > 0.0:
                    log_g_raw = np.nan
                    g_raw = 0.0
                    slack_raw = np.nan
                else:
                    log_g_raw = float(np.mean(np.log(cell_sizes)))
                    g_raw = float(math.exp(log_g_raw))
                    slack_raw = float(log_g_raw + psi - h_cell)
                rows.append({
                    "split": split_name,
                    "policy": policy,
                    "alpha": float(alpha),
                    "audit": int(audit),
                    "profession_id": int(audit_profession_id(meta, audit)),
                    "occupation_name": str(audit_name(meta, audit)),
                    "audit_name": str(audit_name(meta, audit)),
                    "group": int(group),
                    "group_name": str(group_name(meta, group)),
                    "n": int(mask.sum()),
                    "num_labels": int(k),
                    "cell_entropy": h_cell,
                    "error_channel_log_volume": float(psi),
                    "completed_log_geometric_mean_set_size": log_g_plus,
                    "completed_geometric_mean_set_size": g_plus,
                    "completed_product_slack": slack_plus,
                    "completed_excess_log_product_volume": slack_plus,
                    "completed_mean_set_size": float(np.mean(cell_sizes_plus)),
                    "raw_empty_rate": empty_rate,
                    "raw_log_geometric_mean_set_size": log_g_raw,
                    "raw_geometric_mean_set_size": g_raw,
                    "raw_product_slack": slack_raw,
                    "raw_mean_set_size": float(np.mean(cell_sizes)),
                })
    return pd.DataFrame(rows)


def product_slack_gap_table(cell_product: pd.DataFrame, simpson_long: pd.DataFrame) -> pd.DataFrame:
    if cell_product.empty or simpson_long.empty:
        return pd.DataFrame()
    groups = sorted(int(x) for x in cell_product["group"].unique())
    if len(groups) != 2:
        return pd.DataFrame()
    a, b = groups
    rows = []
    simpson_index = simpson_long.set_index(["split", "policy", "alpha", "audit"])
    for (split, policy, alpha, audit), sub in cell_product.groupby(["split", "policy", "alpha", "audit"]):
        ga = sub[sub.group == a]
        gb = sub[sub.group == b]
        if ga.empty or gb.empty:
            continue
        ra = ga.iloc[0]
        rb = gb.iloc[0]
        if (split, policy, alpha, audit) in simpson_index.index:
            sr = simpson_index.loc[(split, policy, alpha, audit)]
            if isinstance(sr, pd.DataFrame):
                sr = sr.iloc[0]
            is_rev = bool(sr.is_simpson_reversal)
            rev_strength = float(sr.reversal_strength)
            local_gap = float(sr.local_gap_delta_z)
            aggregate_gap = float(sr.aggregate_gap_delta)
            burdened_name = str(sr.burdened_group_name)
        else:
            is_rev = False
            rev_strength = 0.0
            local_gap = np.nan
            aggregate_gap = np.nan
            burdened_name = ""
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "audit": int(audit),
            "profession_id": int(ra.profession_id),
            "occupation_name": str(ra.occupation_name),
            "audit_name": str(ra.occupation_name),
            "group_a": int(a),
            "group_a_name": str(ra.group_name),
            "group_b": int(b),
            "group_b_name": str(rb.group_name),
            "delta_log_G_plus_a_minus_b": float(ra.completed_log_geometric_mean_set_size - rb.completed_log_geometric_mean_set_size),
            "delta_entropy_a_minus_b": float(ra.cell_entropy - rb.cell_entropy),
            "delta_slack_plus_a_minus_b": float(ra.completed_product_slack - rb.completed_product_slack),
            "delta_empty_rate_a_minus_b": float(ra.raw_empty_rate - rb.raw_empty_rate),
            "n_a_z": int(ra.n),
            "n_b_z": int(rb.n),
            "aggregate_gap_delta": aggregate_gap,
            "local_gap_delta_z": local_gap,
            "is_simpson_reversal": is_rev,
            "reversal_strength": rev_strength,
            "burdened_group_name": burdened_name,
        })
    return pd.DataFrame(rows)


def active_product_attribution_table(slack_gap: pd.DataFrame, simpson_summary: pd.DataFrame, alpha0: float) -> pd.DataFrame:
    if slack_gap.empty:
        return pd.DataFrame()
    if simpson_summary is None or simpson_summary.empty:
        active_audits = sorted(int(x) for x in slack_gap.audit.unique())
    else:
        active_summary = simpson_summary[simpson_summary.get("num_reversal_alpha", 0).astype(int) > 0].copy()
        if active_summary.empty:
            active_summary = simpson_summary.copy()
        active_audits = [int(x) for x in active_summary.audit.tolist()]
    rows = []
    for audit in active_audits:
        sub = slack_gap[slack_gap.audit == audit].sort_values("alpha").copy()
        if sub.empty:
            continue
        active = sub[sub.is_simpson_reversal].copy()
        if not active.empty:
            idx = active["reversal_strength"].astype(float).idxmax()
            row = sub.loc[idx].copy()
            alpha_rule = "strongest_reversal"
        else:
            idx = int(np.argmin(np.abs(sub.alpha.to_numpy(dtype=float) - float(alpha0))))
            row = sub.iloc[idx].copy()
            alpha_rule = "nearest_alpha0"
        row["attribution_alpha"] = float(row["alpha"])
        row["alpha_selection_rule"] = alpha_rule
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if simpson_summary is not None and not simpson_summary.empty:
        keep_cols = [c for c in ["audit", "num_reversal_alpha", "reversal_fraction", "integrated_reversal_strength", "max_reversal_strength"] if c in simpson_summary.columns]
        out = out.merge(simpson_summary[keep_cols].drop_duplicates("audit"), on="audit", how="left")
    return out.sort_values(["num_reversal_alpha", "reversal_strength"], ascending=[False, False], na_position="last") if "num_reversal_alpha" in out.columns else out


def plot_active_product_attribution(attribution: pd.DataFrame, out_dir: Path, score: str, formats: list[str], top_k: int) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    if attribution.empty:
        return []
    p = bright_palette()
    d = attribution.copy().head(max(int(top_k), 1))
    d = d.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    height = 0.24
    fig_h = max(4.7, 0.48 * len(d) + 2.2)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.barh(y + height, d["delta_log_G_plus_a_minus_b"].astype(float), height=height, color=p["purple"], alpha=0.88, label=r"log-volume gap $\Delta\log G_+$")
    ax.barh(y, d["delta_entropy_a_minus_b"].astype(float), height=height, color=p["geom"], alpha=0.90, label=r"entropy gap $\Delta\widehat H$")
    ax.barh(y - height, d["delta_slack_plus_a_minus_b"].astype(float), height=height, color=p["cell"], alpha=0.90, label=r"excess-slack gap $\Delta S_+$")
    labels = []
    for row in d.itertuples(index=False):
        nrev = int(getattr(row, "num_reversal_alpha", 0)) if pd.notna(getattr(row, "num_reversal_alpha", 0)) else 0
        labels.append(f"{clean_label(row.occupation_name)}\n{nrev} reversals")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#0F172A", linewidth=1.3)
    gap_label = f"{d.group_a_name.iloc[0]} - {d.group_b_name.iloc[0]}" if "group_a_name" in d and not d.empty else "group A - group B"
    ax.set_xlabel(f"cell-wise gap ({gap_label})")
    ax.set_title(f"Appendix: active-cell product attribution ({score_label(score)})", loc="left", fontweight="bold")
    polish_axis(ax, xgrid=True, ygrid=False)
    ax.legend(loc="best", ncol=1, frameon=True, framealpha=0.94)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_appendix_active_cell_product_attribution_{score}", formats)


def plot_slack_gap_heatmap(slack_gap: pd.DataFrame, simpson_summary: pd.DataFrame, out_dir: Path, score: str, formats: list[str], order_mode: str) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if slack_gap.empty:
        return []
    alpha_values = sorted(float(x) for x in slack_gap.alpha.unique())
    order = audit_display_order(simpson_summary, slack_gap, order_mode)
    labels = audit_label_map_from_gap(slack_gap)
    mat = np.full((len(order), len(alpha_values)), np.nan, dtype=float)
    active = np.zeros((len(order), len(alpha_values)), dtype=bool)
    for i, audit in enumerate(order):
        sub = slack_gap[slack_gap.audit == audit]
        by_alpha = {float(r.alpha): r for r in sub.itertuples(index=False)}
        for j, alpha in enumerate(alpha_values):
            row = by_alpha.get(float(alpha))
            if row is None:
                continue
            mat[i, j] = float(row.delta_slack_plus_a_minus_b)
            active[i, j] = bool(row.is_simpson_reversal)
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    fig, ax = plt.subplots(figsize=(13.1, max(5.6, 0.42 * len(order) + 2.0)))
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    add_heatmap_grid(ax, len(order), len(alpha_values))
    for i in range(len(order)):
        for j in range(len(alpha_values)):
            if active[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="#111827", linewidth=1.25)
                ax.add_patch(rect)
    xticks, xticklabels = alpha_tick_positions(alpha_values, max_ticks=9)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([clean_label(labels.get(a, str(a))) for a in order])
    ax.set_xlabel("miscoverage level")
    ax.set_ylabel("occupation")
    ax.set_title(f"Appendix: excess-slack gap with Simpson cells ({score_label(score)})", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.018)
    cbar.set_label(r"excess-slack gap $\Delta S_+$")
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_appendix_slack_gap_heatmap_{score}", formats)


def plot_entropy_slack_scatter(attribution: pd.DataFrame, out_dir: Path, score: str, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    if attribution.empty:
        return []
    p = bright_palette()
    d = attribution.copy()
    if "num_reversal_alpha" not in d.columns:
        d["num_reversal_alpha"] = 0
    fig, ax = plt.subplots(figsize=(8.7, 6.4))
    sizes = 70.0 + 32.0 * d["num_reversal_alpha"].fillna(0).astype(float).to_numpy()
    colors = []
    for row in d.itertuples(index=False):
        burden = str(getattr(row, "burdened_group_name", ""))
        if burden:
            colors.append(group_color(burden, 0))
        else:
            colors.append(p["gray"])
    ax.scatter(
        d["delta_entropy_a_minus_b"].astype(float),
        d["delta_slack_plus_a_minus_b"].astype(float),
        s=sizes,
        c=colors,
        alpha=0.84,
        edgecolor="white",
        linewidth=1.0,
    )
    for row in d.itertuples(index=False):
        nrev = int(getattr(row, "num_reversal_alpha", 0)) if pd.notna(getattr(row, "num_reversal_alpha", 0)) else 0
        if nrev > 0:
            ax.annotate(clean_label(row.occupation_name), (float(row.delta_entropy_a_minus_b), float(row.delta_slack_plus_a_minus_b)), xytext=(4, 4), textcoords="offset points", fontsize=9.8, color="#0F172A")
    ax.axhline(0.0, color="#0F172A", linewidth=1.1)
    ax.axvline(0.0, color="#0F172A", linewidth=1.1)
    gap_label = f"{d.group_a_name.iloc[0]} - {d.group_b_name.iloc[0]}" if "group_a_name" in d and not d.empty else "group A - group B"
    ax.set_xlabel(rf"entropy gap $\Delta\widehat H$ ({gap_label})")
    ax.set_ylabel(rf"excess-slack gap $\Delta S_+$ ({gap_label})")
    ax.set_title(f"Appendix: intrinsic difficulty versus score-geometry slack ({score_label(score)})", loc="left", fontweight="bold")
    polish_axis(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_appendix_entropy_slack_scatter_{score}", formats)

def plot_occupation_composition_global(raw_repair: dict[str, np.ndarray], meta: dict[str, Any], out_root: Path, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    z = raw_repair["z"]
    g = raw_repair["g"]
    audits = sorted(int(x) for x in np.unique(z))
    groups = sorted(int(x) for x in np.unique(g))
    if len(groups) != 2:
        return []
    p = bright_palette()
    props = np.zeros((len(groups), len(audits)), dtype=float)
    for i, group in enumerate(groups):
        denom = max(int(np.sum(g == group)), 1)
        for j, audit in enumerate(audits):
            props[i, j] = float(np.sum((g == group) & (z == audit))) / float(denom)

    x = np.arange(len(audits))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    for i, group in enumerate(groups):
        gname = group_name(meta, group)
        offset = (i - 0.5) * width
        ax.bar(x + offset, props[i], width=width, color=group_color(gname, i), alpha=0.88, label=gname)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(audit_name(meta, a)) for a in audits], rotation=30, ha="right")
    ax.set_ylabel("occupation share within group")
    ax.set_xlabel("occupation")
    ax.set_title("Appendix: occupation composition by group on D_repair", loc="left", fontweight="bold")
    polish_axis(ax, xgrid=False, ygrid=True)
    ax.legend(loc="upper right", ncol=2)
    figures_dir = mkdir(out_root / "figures")
    return save_figure(fig, figures_dir / "fig_appendix_occupation_composition", formats)

def run_score(score: str, args: argparse.Namespace, raw: dict[str, dict[str, np.ndarray]], meta: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Run one score family using D_repair only.

    The key invariant is:
        D_repair -> thresholds, aggregate gaps, cell gaps, active Simpson cells.

    No D_cal information is read or used. Therefore the generated active-cell
    tables are safe to feed into a downstream score-repair algorithm.
    """
    root = mkdir(out_root / score)
    tables_dir = mkdir(root / "tables")
    figures_dir = mkdir(root / "figures")
    arrays_dir = mkdir(root / "arrays")

    data: dict[str, dict[str, np.ndarray]] = {"repair": {**raw["repair"]}}
    data["repair"]["S"] = scores_from_probs(
        raw["repair"]["probs"],
        score=score,
        saps_lambda=args.saps_lambda,
        raps_lambda=args.raps_lambda,
        raps_k_reg=args.raps_k_reg,
        randomize=args.score_randomize,
        seed=args.score_random_seed,
        temperature=args.score_temperature,
    )
    np.save(arrays_dir / f"scores_repair_{score}.npy", data["repair"]["S"])
    np.save(arrays_dir / f"true_scores_repair_{score}.npy", true_scores(data["repair"]["S"], data["repair"]["y"]))

    thresholds = build_thresholds(data["repair"], args.alpha_grid_values)
    curve_repair = curve_table(data["repair"], thresholds, args.alpha_grid_values, meta, "repair")
    curve_all = curve_repair.copy()
    aggregate_gaps = aggregate_gap_table(curve_all)
    cell_repair = cell_curve_table(data["repair"], thresholds, args.alpha_grid_values, meta, "repair", args.diagnostic_policy)
    entropy_repair = entropy_floor_table(data["repair"], thresholds, args.alpha_grid_values, meta, "repair", args.diagnostic_policy)
    cell_product_repair = cell_product_slack_table(data["repair"], thresholds, args.alpha_grid_values, meta, "repair", args.diagnostic_policy)
    simpson_long, simpson_summary = simpson_gap_curve(
        aggregate_gaps[(aggregate_gaps.split == "repair") & (aggregate_gaps.policy == args.diagnostic_policy)],
        cell_repair,
        n_min=args.n_min,
        tau_agg=args.tau_agg,
        tau_loc=args.tau_loc,
        alpha0=args.alpha0,
    )

    if not simpson_long.empty:
        active_cells = simpson_long[simpson_long.is_simpson_reversal].copy()
    else:
        active_cells = simpson_long.copy()
    product_slack_gaps = product_slack_gap_table(cell_product_repair, simpson_long)
    active_product_attribution = active_product_attribution_table(product_slack_gaps, simpson_summary, args.alpha0)

    tables = {
        "aggregate_curve_repair_only.csv": curve_all,
        "aggregate_curve.csv": curve_all,
        "aggregate_gap_curve.csv": aggregate_gaps,
        "cell_curve_repair.csv": cell_repair,
        "entropy_floor_diagnostic.csv": entropy_repair,
        "product_uncertainty_diagnostic.csv": entropy_repair,
        "cell_product_slack_curve_long.csv": cell_product_repair,
        "product_slack_gap_curve_long.csv": product_slack_gaps,
        "active_cell_product_attribution.csv": active_product_attribution,
        "simpson_gap_curve_long.csv": simpson_long,
        "simpson_curve_summary.csv": simpson_summary,
        "active_simpson_curve_cells.csv": active_cells,
    }
    if args.write_sample_tables:
        repair_samples = sample_level_table(data["repair"], thresholds, args.alpha0, args.diagnostic_policy, meta, "repair")
        if not active_cells.empty:
            active_audits = sorted(active_cells.audit.unique())
            active_samples = repair_samples[repair_samples.audit.isin(active_audits)].copy()
        else:
            active_samples = repair_samples.iloc[0:0].copy()
        tables["repair_sample_level_at_alpha0.csv"] = repair_samples
        tables["active_reversal_samples.csv"] = active_samples
    for name, df in tables.items():
        df.to_csv(tables_dir / name, index=False)

    formats = [x.strip().lower() for x in args.figure_formats.split(",") if x.strip()]
    figure_paths = []
    figure_paths += plot_main_figure(curve_all, simpson_long, simpson_summary, figures_dir, score, formats, args.occupation_order)
    figure_paths += plot_occupation_curve_grid(cell_repair, curve_all, simpson_long, simpson_summary, figures_dir, score, formats, args.occupation_order)
    figure_paths += plot_gap_curve_grid(simpson_long, simpson_summary, figures_dir, score, formats, args.occupation_order)
    figure_paths += plot_entropy_floor_figure(entropy_repair, figures_dir, score, formats)
    figure_paths += plot_product_uncertainty_gplus_figure(entropy_repair, figures_dir, score, formats)
    figure_paths += plot_empty_set_rate_figure(entropy_repair, figures_dir, score, formats)
    figure_paths += plot_active_product_attribution(active_product_attribution, figures_dir, score, formats, args.top_k_cells)
    figure_paths += plot_slack_gap_heatmap(product_slack_gaps, simpson_summary, figures_dir, score, formats, args.occupation_order)
    figure_paths += plot_entropy_slack_scatter(active_product_attribution, figures_dir, score, formats)

    manifest = {
        "score": score,
        "score_root": str(root),
        "analysis_mode": "repair_only",
        "threshold_source_split": "repair",
        "diagnostic_split": "repair",
        "diagnostic_policy": args.diagnostic_policy,
        "alpha0": float(args.alpha0),
        "alpha_grid": args.alpha_grid_values,
        "repair_safety": {
            "safe_for_repair_selection": True,
            "rule": "This run used D_repair only. No D_cal threshold or label information was read.",
        },
        "n_active_alpha_cell_reversals": int(active_cells.shape[0]) if not active_cells.empty else 0,
        "n_active_occupations": int(active_cells.audit.nunique()) if not active_cells.empty else 0,
        "top_diagnostic_cells": simpson_summary.head(args.top_k_cells).to_dict("records") if not simpson_summary.empty else [],
        "score_parameters": {
            "score_temperature": float(args.score_temperature),
            "score_randomize": bool(args.score_randomize),
            "score_random_seed": int(args.score_random_seed),
            "saps_lambda": float(args.saps_lambda),
            "raps_lambda": float(args.raps_lambda),
            "raps_k_reg": int(args.raps_k_reg),
        },
        "tables": sorted(tables.keys()),
        "figures": figure_paths,
    }
    jdump(manifest, root / "manifest.json")
    return manifest

def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Repair-only BioBias curve and Simpson analysis using D_repair only.")
    p.add_argument("--seed-dir", type=Path, default=None, help="Path to outputs_biobias_train/<run>/seed_X")
    p.add_argument("--run-root", type=Path, default=None, help="Path to outputs_biobias_train/<run>; use with --seed")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prob-dir", type=Path, default=None, help="Path directly containing probability artifacts")
    p.add_argument("--prob-zip", type=Path, default=None, help="Zip file containing a probabilities directory")
    p.add_argument("--meta-json", type=Path, default=None, help="Optional split_meta.json or label_maps.json")
    p.add_argument("--label-names", default=None, help="Optional comma-separated occupation names")
    p.add_argument("--group-names", default=None, help="Optional comma-separated group names")
    p.add_argument("--output-root", type=Path, default=None, help="Default: <prob_dir>/../curve_diagnostics_repair_only")
    p.add_argument("--clean-output", action="store_true", help="Remove output root before writing new analysis artifacts")
    p.add_argument("--score", choices=["simple", "aps", "saps", "raps", "all"], default="all")
    p.add_argument("--saps-lambda", type=float, default=0.2)
    p.add_argument("--raps-lambda", type=float, default=0.2)
    p.add_argument("--raps-k-reg", type=int, default=5)
    p.add_argument("--score-temperature", type=float, default=1.0)
    p.add_argument("--score-randomize", action="store_true")
    p.add_argument("--score-random-seed", type=int, default=0)
    p.add_argument("--alpha0", type=float, default=0.10)
    p.add_argument("--alpha-grid", default="dense_safe")
    p.add_argument("--diagnostic-policy", choices=["pooled", "groupwise"], default="groupwise")
    p.add_argument("--n-min", type=int, default=40)
    p.add_argument("--tau-agg", type=float, default=0.005)
    p.add_argument("--tau-loc", type=float, default=0.02)
    p.add_argument("--top-k-cells", type=int, default=6)
    p.add_argument("--occupation-order", choices=["evidence", "label"], default="evidence")
    p.add_argument("--write-sample-tables", action="store_true", help="Optionally write large sample-level trace tables")
    p.add_argument("--figure-formats", default="pdf,png,svg")
    return p


def main() -> None:
    args = args_parser().parse_args()
    args.alpha_grid_values = parse_grid(args.alpha_grid)
    if max(args.alpha_grid_values) > 0.1000001:
        print("Warning: alpha-grid extends beyond 0.10. Raw conformal sets may become empty more often; interpret geometric entropy diagnostics with care.")
    if args.alpha0 not in args.alpha_grid_values:
        args.alpha_grid_values = sorted(args.alpha_grid_values + [float(args.alpha0)])

    raw, meta, prob_dir, tmp_dir = load_artifact(args)
    if args.output_root is not None:
        out_root = args.output_root.expanduser().resolve()
    else:
        out_root = prob_dir.parent / "curve_diagnostics_repair_only"
    if args.clean_output and out_root.exists():
        shutil.rmtree(out_root)
    mkdir(out_root)

    if "selected_professions" in meta:
        pd.DataFrame(meta["selected_professions"]).to_csv(out_root / "resolved_label_maps.csv", index=False)
        jdump({
            "selected_professions": meta["selected_professions"],
            "gender_names": meta.get("gender_names", []),
            "meta_source": meta.get("meta_source", "unknown"),
        }, out_root / "resolved_label_maps.json")

    scores = ["simple", "aps", "saps", "raps"] if args.score == "all" else [args.score]
    manifests = []
    global_figures: list[str] = []
    try:
        for score in scores:
            print(f"Analyzing repair-only curve diagnostics for score={score} using D_repair only ...")
            manifest = run_score(score, args, raw, meta, out_root)
            manifests.append(manifest)
            print(
                f"  saved: {manifest['score_root']} | "
                f"active alpha-cell reversals: {manifest['n_active_alpha_cell_reversals']} | "
                f"active occupations: {manifest['n_active_occupations']}"
            )
        formats = [x.strip().lower() for x in args.figure_formats.split(",") if x.strip()]
        global_figures += plot_support_counts_global(raw["repair"], meta, out_root, formats)
        global_figures += plot_occupation_composition_global(raw["repair"], meta, out_root, formats)
        if len(scores) > 1:
            global_figures += plot_score_robustness_heatmaps(out_root, scores, formats, args.occupation_order)
        jdump({
            "probability_dir": str(prob_dir),
            "output_root": str(out_root),
            "analysis_mode": "repair_only",
            "threshold_source_split": "repair",
            "diagnostic_split": "repair",
            "repair_safety_rule": "This script uses D_repair only and does not read D_cal.",
            "meta_source": meta.get("meta_source", "split_meta_json_or_user_meta"),
            "scores": manifests,
            "global_figures": global_figures,
            "kept_figure_policy": {
                "per_score": [
                    "fig_main_biobias_curve_simpson_<score>",
                    "fig_appendix_occupation_curve_grid_<score>",
                    "fig_appendix_gap_curve_grid_<score>",
                    "fig_product_uncertainty_<score>",
                    "fig_product_uncertainty_gplus_<score>",
                    "fig_appendix_active_cell_product_attribution_<score>",
                    "fig_appendix_slack_gap_heatmap_<score>",
                    "fig_appendix_entropy_slack_scatter_<score>",
                ],
                "global": [
                    "fig_appendix_support_counts",
                    "fig_appendix_occupation_composition",
                    "fig_appendix_score_robustness_heatmaps when score=all",
                ],
            },
        }, out_root / "analysis_manifest.json")
        print(f"Done. Repair-only curve diagnostics saved to: {out_root}")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
