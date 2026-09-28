#!/usr/bin/env python3
"""
BioBias deployment-safe learned-propensity adjusted curve.

This script trains a label-free balancing model on D_repair and evaluates the
resulting adjusted set-size curves on D_repair as a diagnostic stage.
It does NOT repair nonconformity scores, thresholds, or prediction sets.

Key idea
--------
We learn

    e(H) = P(G = positive_group | H)

on D_repair, where H is deployment-safe: it is constructed only from the frozen
occupation classifier probabilities p_hat(.|x), e.g. predicted occupation,
entropy, top probabilities, margins.  H does not include the true occupation Y.

The learned e(H) gives inverse-propensity or overlap weights.  We then estimate
weighted set-size curves using the original conformal prediction sets:

    V_g^w(alpha) = sum_{i:G_i=g} w_i |C_alpha(X_i)| / sum_{i:G_i=g} w_i.

True occupation labels are used only for post-hoc audit grids V_{g,z}^w(alpha),
not for constructing H or weights.

Default output:

    <seed_dir>/curve_adjusted_learned_propensity/

If that directory exists and is non-empty, _02, _03, ... is appended to avoid
overwriting prior results.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Stable module alias so custom propensity models pickle/unpickle correctly even
# when this file is executed as a script.
sys.modules.setdefault("biobias_adjusted_curve_learned_propensity", sys.modules[__name__])

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scikit-learn. Install with: pip install scikit-learn") from exc


DEFAULT_OUTPUT_DIRNAME = "curve_adjusted_learned_propensity"
BIOBIAS_GENDER_ID_TO_NAME = {0: "Male", 1: "Female"}
DEFAULT_SCORE_PROFILE_GRID = "0.50,0.60,0.70,0.80,0.90,0.95,0.98"
DEFAULT_SIMPLE_SCORE_GRID = "0.05,0.10,0.20,0.30,0.40,0.50,0.70,0.90"
DEFAULT_SAPS_RAPS_SCORE_GRID = "0.50,0.70,0.90,1.10,1.30,1.60,2.00,2.50"


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def unique_output_path(path: Path) -> Path:
    if not path_has_content(path):
        return path
    for k in range(2, 1000):
        cand = path.with_name(f"{path.name}_{k:02d}")
        if not path_has_content(cand):
            return cand
    raise RuntimeError(f"Could not find non-overwriting output directory near {path}")


def jdump(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [conv(v) for v in x]
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

    path.write_text(json.dumps(conv(obj), indent=2), encoding="utf-8")


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            # Backward-compatible unpickling for bundles created by running this
            # file as a script, where custom classes may have been stored under
            # the __main__ module path.
            import __main__
            if "TorchMLPPropensity" in str(exc):
                setattr(__main__, "TorchMLPPropensity", TorchMLPPropensity)
                f.seek(0)
                return pickle.load(f)
            raise


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
        raise ValueError("--alpha-grid must contain values in (0,1), or use dense_safe/dense_010/dense_012.")
    return [float(x) for x in values]


def setup_plotting() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 360,
        "font.size": 12.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 12.0,
        "legend.fontsize": 9.8,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#F8FAFC",
        "figure.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#A8B3C7",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "grid.alpha": 0.90,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def colors() -> dict[str, str]:
    return {
        "male": "#0077FF",
        "female": "#FF4FA3",
        "raw": "#64748B",
        "adjusted": "#7C3AED",
        "gold": "#F59E0B",
        "green": "#10B981",
        "red": "#E11D48",
        "dark": "#0F172A",
        "shade": "#FDE68A",
    }


def group_color(name: str, idx: int) -> str:
    c = colors()
    low = str(name).lower()
    if "male" in low and "female" not in low:
        return c["male"]
    if "female" in low:
        return c["female"]
    return [c["male"], c["female"], c["adjusted"], c["gold"]][idx % 4]


def save_figure(fig: Any, stem: Path, formats: list[str]) -> list[str]:
    paths = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
        plt.close("all")
    except Exception:
        pass
    return paths


# -----------------------------------------------------------------------------
# Data loading and names
# -----------------------------------------------------------------------------


def resolve_seed_dir(args: argparse.Namespace) -> Path:
    if args.seed_dir is not None:
        return args.seed_dir.expanduser().resolve()
    if args.run_root is not None:
        if args.seed is None:
            raise FileNotFoundError("Provide --seed when using --run-root.")
        run_root = args.run_root.expanduser().resolve()
        return run_root if run_root.name.startswith("seed_") else run_root / f"seed_{int(args.seed)}"
    if args.prob_dir is not None:
        return args.prob_dir.expanduser().resolve().parent
    raise FileNotFoundError("Use --seed-dir, --run-root with --seed, or --prob-dir.")


def resolve_prob_dir(args: argparse.Namespace, seed_dir: Path) -> Path:
    return args.prob_dir.expanduser().resolve() if args.prob_dir is not None else seed_dir / "probabilities"


def load_json_first(paths: list[Path]) -> dict[str, Any]:
    for p in paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def load_meta(prob_dir: Path, seed_dir: Path) -> dict[str, Any]:
    meta = load_json_first([
        prob_dir / "label_maps.json",
        seed_dir / "label_maps.json",
        prob_dir / "split_meta.json",
        seed_dir / "split_meta.json",
    ])
    if "group_names" not in meta and "gender_names" in meta:
        meta["group_names"] = meta["gender_names"]
    if "gender_names" not in meta and "group_names" in meta:
        meta["gender_names"] = meta["group_names"]
    return meta


def np_load_optional(prob_dir: Path, key: str, split: str, allow_pickle: bool = True) -> np.ndarray | None:
    p = prob_dir / f"{key}_{split}.npy"
    if p.exists():
        return np.load(p, allow_pickle=allow_pickle)
    return None


def load_split(prob_dir: Path, split: str) -> dict[str, np.ndarray]:
    required = [f"probs_{split}.npy", f"y_{split}.npy", f"g_{split}.npy"]
    missing = [x for x in required if not (prob_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {prob_dir}: {missing}")
    d: dict[str, np.ndarray] = {
        "probs": np.load(prob_dir / f"probs_{split}.npy"),
        "y": np.load(prob_dir / f"y_{split}.npy"),
        "g": np.load(prob_dir / f"g_{split}.npy"),
    }
    # z is the post-hoc audit occupation. In this artifact z_equals_y, but it is
    # never used for deployment-safe H construction.
    z = np_load_optional(prob_dir, "z", split)
    d["z"] = np.asarray(z if z is not None else d["y"], dtype=int)
    for key in [
        "idx", "artifact_index", "sample_id", "source_split", "source_index",
        "profession_id", "occupation_name", "group_name", "z_profession_id", "z_occupation_name",
    ]:
        arr = np_load_optional(prob_dir, key, split)
        if arr is not None:
            d[key] = arr
    return d


def load_splits(prob_dir: Path, splits: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {s: load_split(prob_dir, s) for s in splits}


def group_name(meta: dict[str, Any], g: int) -> str:
    names = meta.get("gender_names", []) or meta.get("group_names", [])
    if 0 <= int(g) < len(names):
        return str(names[int(g)])
    return BIOBIAS_GENDER_ID_TO_NAME.get(int(g), f"group_{int(g)}")


def occupation_name(meta: dict[str, Any], z: int) -> str:
    names = meta.get("occupation_names", []) or meta.get("label_names", []) or meta.get("audit_names", [])
    if 0 <= int(z) < len(names):
        return str(names[int(z)])
    mapping = meta.get("compact_label_to_occupation_name", {})
    return str(mapping.get(str(int(z)), f"occupation_{int(z)}"))


def profession_id(meta: dict[str, Any], z: int) -> int:
    mapping = meta.get("compact_label_to_profession_id", {}) or meta.get("new_label_to_old_profession_id", {})
    return int(mapping.get(str(int(z)), int(z)))


def positive_group_id(meta: dict[str, Any], groups: np.ndarray, requested: str) -> int:
    req = str(requested).strip().lower()
    for g in sorted(np.unique(groups)):
        if group_name(meta, int(g)).strip().lower() == req:
            return int(g)
    if req in {"male", "m"}:
        for g in sorted(np.unique(groups)):
            nm = group_name(meta, int(g)).lower()
            if "male" in nm and "female" not in nm:
                return int(g)
    return int(sorted(np.unique(groups))[0])


# -----------------------------------------------------------------------------
# Scores and conformal prediction sets
# -----------------------------------------------------------------------------


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


def build_thresholds(source: dict[str, np.ndarray], alphas: list[float]) -> dict[str, dict[float, float | dict[int, float]]]:
    ts = true_scores(source["S"], source["y"])
    return {
        "pooled": {float(a): qhat(ts, float(a)) for a in alphas},
        "groupwise": {float(a): group_qhat(ts, source["g"], float(a)) for a in alphas},
    }


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
    return (scores <= thr[:, None]).sum(axis=1).astype(float)


def covered(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, threshold: float | dict[int, float]) -> np.ndarray:
    thr = thresholds_for(groups, threshold)
    return (true_scores(scores, y) <= thr).astype(float)


# -----------------------------------------------------------------------------
# Deployment-safe H and learned propensity weights
# -----------------------------------------------------------------------------


def _optional_length_features(split: dict[str, np.ndarray] | None, n: int) -> tuple[np.ndarray, list[str]]:
    """Return deployment-safe length features when exported; otherwise a missingness indicator.

    These features never use the true occupation/label.  If the original exporter
    did not save text/token lengths, the function still returns a fixed two-column
    block so that *_len modes remain shape-stable across repair/cal/test.
    """
    keys = [
        "text_len", "text_length", "char_len", "num_chars",
        "token_len", "token_length", "num_tokens", "input_length",
        "attention_mask_sum", "seq_len",
    ]
    arr = None
    if split is not None:
        for key in keys:
            if key in split:
                cand = np.asarray(split[key])
                if cand.shape[0] == n and np.issubdtype(cand.dtype, np.number):
                    arr = cand.astype(float)
                    break
    if arr is None:
        return np.c_[np.zeros(n), np.ones(n)], ["length_log1p_missing0", "length_missing_indicator"]
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    arr = np.where(np.isfinite(arr), arr, med)
    return np.c_[np.log1p(np.maximum(arr, 0.0)), np.zeros(n)], ["length_log1p", "length_missing_indicator"]



def _parse_float_list(text: str | list[float] | tuple[float, ...], *, name: str, lower: float | None = None, upper: float | None = None) -> list[float]:
    if isinstance(text, (list, tuple, np.ndarray)):
        vals = [float(x) for x in text]
    else:
        vals = [float(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} must contain at least one numeric value")
    vals = sorted(dict.fromkeys(vals))
    if lower is not None and any(v <= lower for v in vals):
        raise ValueError(f"{name} values must be > {lower}")
    if upper is not None and any(v >= upper for v in vals):
        raise ValueError(f"{name} values must be < {upper}")
    return vals


def _score_count_fingerprint(scores: np.ndarray, grid: list[float], prefix: str) -> tuple[np.ndarray, list[str]]:
    """Label-free set-size fingerprint: number of candidate labels below fixed score cutoffs."""
    S = np.asarray(scores, dtype=float)
    blocks = []
    names = []
    for q in grid:
        blocks.append((S <= float(q)).sum(axis=1).astype(float))
        names.append(f"{prefix}_count_score_le_{float(q):.3g}")
    return np.column_stack(blocks), names


def probability_features(
    probs: np.ndarray,
    mode: str,
    top_k: int = 3,
    split: dict[str, np.ndarray] | None = None,
    score_profile_grid: str | list[float] | tuple[float, ...] = DEFAULT_SCORE_PROFILE_GRID,
    simple_score_grid: str | list[float] | tuple[float, ...] = DEFAULT_SIMPLE_SCORE_GRID,
    saps_raps_score_grid: str | list[float] | tuple[float, ...] = DEFAULT_SAPS_RAPS_SCORE_GRID,
) -> tuple[np.ndarray, list[str]]:
    """Construct deployment-safe H from p_hat(.|x), optionally plus safe lengths.

    Safety rule: this function must never use true occupation Y/Z, conformal true
    scores s(x,Y), true-label rank, or realized prediction-set size.  It only uses
    quantities visible at prediction time: frozen-model probabilities, their
    label-free score profiles, and optionally exported text/token lengths.

    The ``safe_score_profile`` mode enriches the balancing score with
    classifier-induced prediction geometry:
    sorted probabilities, cumulative top-mass, and label-free counts of how many
    candidate labels would pass fixed score cutoffs.  Thus e(H) remains a standard
    deployment-safe propensity/balancing score, while H is more adequate for
    curve/set-size comparability.
    """
    p = np.asarray(probs, dtype=float)
    n, k = p.shape
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    pred = order[:, 0]
    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1] if k >= 2 else np.zeros(n)
    p3 = sorted_p[:, 2] if k >= 3 else np.zeros(n)
    entropy = -np.sum(p * np.log(p), axis=1)
    entropy_norm = entropy / max(math.log(k), eps)
    margin12 = p1 - p2
    margin23 = p2 - p3
    top_mass3 = sorted_p[:, : min(3, k)].sum(axis=1)

    blocks: list[np.ndarray] = []
    names: list[str] = []
    mode = str(mode).lower()
    base_mode = mode[:-4] if mode.endswith("_len") else mode

    if base_mode in {"safe_main", "predicted", "predicted_difficulty"}:
        one = np.zeros((n, k), dtype=float)
        one[np.arange(n), pred] = 1.0
        blocks.append(one)
        names += [f"pred_label_{j}" for j in range(k)]
        cont = np.c_[p1, p2, p3, entropy_norm, margin12, margin23, top_mass3]
        blocks.append(cont)
        names += ["p_top1", "p_top2", "p_top3", "entropy_norm", "margin12", "margin23", "top3_mass"]
    elif base_mode in {"safe_topk", "topk"}:
        kk = max(1, min(int(top_k), k))
        for r in range(kk):
            one = np.zeros((n, k), dtype=float)
            one[np.arange(n), order[:, r]] = 1.0
            blocks.append(one)
            names += [f"pred_top{r+1}_label_{j}" for j in range(k)]
        top_probs = sorted_p[:, :kk]
        blocks.append(top_probs)
        names += [f"p_top{r+1}" for r in range(kk)]
        cont = np.c_[entropy_norm, margin12, margin23, top_mass3]
        blocks.append(cont)
        names += ["entropy_norm", "margin12", "margin23", "top3_mass"]
    elif base_mode in {"safe_probs", "prob_vector", "full_probs"}:
        blocks.append(p.copy())
        names += [f"prob_{j}" for j in range(k)]
        cont = np.c_[p1, p2, p3, entropy_norm, margin12, margin23, top_mass3]
        blocks.append(cont)
        names += ["p_top1", "p_top2", "p_top3", "entropy_norm", "margin12", "margin23", "top3_mass"]
    elif base_mode in {"safe_score_profile", "score_profile", "safe_curve_profile"}:
        kk = max(1, min(int(top_k), k))
        # Identity/composition information: which labels dominate the frozen classifier.
        for r in range(kk):
            one = np.zeros((n, k), dtype=float)
            one[np.arange(n), order[:, r]] = 1.0
            blocks.append(one)
            names += [f"top{r+1}_label_{j}" for j in range(k)]
        # Full probability vector and rank-only probability geometry.
        blocks.append(p.copy())
        names += [f"prob_{j}" for j in range(k)]
        blocks.append(sorted_p.copy())
        names += [f"p_sorted_{r+1}" for r in range(k)]
        cumsum_p = np.cumsum(sorted_p, axis=1)
        blocks.append(cumsum_p)
        names += [f"p_cumsum_top{r+1}" for r in range(k)]
        # Smooth difficulty/ambiguity summaries.
        top_masses = np.column_stack([sorted_p[:, :m].sum(axis=1) for m in range(1, min(5, k) + 1)])
        eff_num = np.exp(entropy)
        l2_conc = np.sum(p * p, axis=1)
        cont = np.c_[p1, p2, p3, entropy_norm, eff_num, l2_conc, margin12, margin23, top_mass3, top_masses]
        blocks.append(cont)
        names += ["p_top1", "p_top2", "p_top3", "entropy_norm", "effective_num_labels", "prob_l2_concentration", "margin12", "margin23", "top3_mass"]
        names += [f"top{m}_mass" for m in range(1, min(5, k) + 1)]
        # Cumulative-mass fingerprints: number of labels needed to reach a fixed mass.
        mass_grid = _parse_float_list(score_profile_grid, name="score_profile_grid", lower=0.0, upper=1.0)
        mass_counts = []
        for gam in mass_grid:
            mass_counts.append((np.argmax(cumsum_p >= float(gam), axis=1) + 1).astype(float))
            names.append(f"num_labels_to_mass_{float(gam):.3g}")
        blocks.append(np.column_stack(mass_counts))
        # Label-free candidate-set fingerprints for standard score geometries.  These
        # are fixed cutoffs, not calibration quantiles, so they do not use D_cal/test labels.
        simple_grid = _parse_float_list(simple_score_grid, name="simple_score_grid", lower=0.0, upper=None)
        sraps_grid = _parse_float_list(saps_raps_score_grid, name="saps_raps_score_grid", lower=0.0, upper=None)
        S_simple = scores_from_probs(p, "simple", 0.2, 0.2, 5, False, 0, 1.0)
        B, bn = _score_count_fingerprint(S_simple, simple_grid, "simple")
        blocks.append(B); names += bn
        S_saps = scores_from_probs(p, "saps", 0.2, 0.2, 5, False, 0, 1.0)
        B, bn = _score_count_fingerprint(S_saps, sraps_grid, "saps")
        blocks.append(B); names += bn
        S_raps = scores_from_probs(p, "raps", 0.2, 0.2, 5, False, 0, 1.0)
        B, bn = _score_count_fingerprint(S_raps, sraps_grid, "raps")
        blocks.append(B); names += bn
    else:
        raise ValueError("--feature-mode must be safe_main, safe_main_len, safe_topk, safe_probs, safe_probs_len, safe_score_profile, or safe_score_profile_len")

    if mode.endswith("_len"):
        lf, ln = _optional_length_features(split, n)
        blocks.append(lf)
        names += ln

    X = np.c_[tuple(blocks)]
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    if X.shape[1] != len(names):
        raise RuntimeError(
            f"Feature-name mismatch for mode={mode}: X has {X.shape[1]} columns but names has {len(names)} entries."
        )
    return X, names

def parse_hidden_layers(text: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(text, tuple):
        return tuple(int(x) for x in text)
    if isinstance(text, list):
        return tuple(int(x) for x in text)
    vals = [int(x.strip()) for x in str(text).replace("x", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("--mlp-hidden-layers must contain at least one positive integer, e.g. 64,32")
    if any(v <= 0 for v in vals):
        raise ValueError("--mlp-hidden-layers must contain positive integers")
    return tuple(vals)



class TorchMLPPropensity:
    """Small PyTorch MLP with a scikit-learn-like predict_proba interface.

    It is intentionally small and regularized.  The model is only used to learn a
    deployment-safe balancing score e(H); it does not see true occupations or set
    sizes.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 4096,
        patience: int = 10,
        val_fraction: float = 0.15,
        seed: int = 0,
        device: str = "auto",
        num_threads: int = 1,
    ) -> None:
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.device_requested = str(device)
        self.num_threads = int(num_threads)
        self.device_ = "cpu"
        self.model_ = None
        self.input_dim_ = None
        self.training_history_ = []

    def _make_model(self, input_dim: int):
        import torch
        layers = []
        prev = int(input_dim)
        for h in self.hidden_layers:
            layers.append(torch.nn.Linear(prev, int(h)))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=0.05))
            prev = int(h)
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.num_threads and self.num_threads > 0:
            torch.set_num_threads(int(self.num_threads))
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.input_dim_ = int(X.shape[1])
        if self.device_requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_requested
        self.device_ = device
        idx = np.arange(len(y))
        if self.val_fraction > 0 and len(np.unique(y)) == 2 and len(y) > 100:
            tr_idx, val_idx = train_test_split(idx, test_size=self.val_fraction, random_state=self.seed, stratify=y.ravel())
        else:
            tr_idx, val_idx = idx, idx
        model = self._make_model(self.input_dim_).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)
        X_val = torch.from_numpy(X[val_idx]).to(device)
        y_val = torch.from_numpy(y[val_idx]).to(device)
        best_loss = float("inf")
        best_state = None
        bad = 0
        self.training_history_ = []
        for ep in range(max(1, self.epochs)):
            model.train()
            total = 0.0
            n_seen = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(xb)
                n_seen += len(xb)
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val), y_val).detach().cpu())
            self.training_history_.append({"epoch": ep + 1, "train_loss": total / max(n_seen, 1), "val_loss": val_loss})
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.to("cpu")
        self.device_ = "cpu"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self.model_ is None:
            raise RuntimeError("TorchMLPPropensity is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X)).detach().cpu().numpy().reshape(-1)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return np.c_[1.0 - p1, p1]


# Make pickles portable when the training script is executed as __main__.
TorchMLPPropensity.__module__ = "biobias_adjusted_curve_learned_propensity"


def build_propensity_estimator(model_type: str, args: argparse.Namespace, seed: int) -> Any:
    mt = str(model_type).lower()
    if mt == "logistic":
        return LogisticRegression(
            C=float(args.logistic_c),
            max_iter=2000,
            solver="lbfgs",
            class_weight=None,
            random_state=int(seed),
        )
    if mt == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=int(args.histgb_max_iter),
            learning_rate=float(args.histgb_learning_rate),
            l2_regularization=float(args.histgb_l2),
            max_leaf_nodes=int(args.histgb_max_leaf_nodes),
            random_state=int(seed),
        )
    if mt == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=parse_hidden_layers(args.mlp_hidden_layers),
            activation="relu",
            solver="adam",
            alpha=float(args.mlp_alpha),
            learning_rate_init=float(args.mlp_lr),
            batch_size="auto",
            max_iter=int(args.mlp_max_iter),
            early_stopping=bool(args.mlp_early_stopping),
            validation_fraction=float(args.mlp_validation_fraction),
            n_iter_no_change=int(args.mlp_n_iter_no_change),
            random_state=int(seed),
        )
    if mt == "torch_mlp":
        return TorchMLPPropensity(
            hidden_layers=parse_hidden_layers(args.mlp_hidden_layers),
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_alpha),
            epochs=int(args.torch_mlp_epochs),
            batch_size=int(args.torch_mlp_batch_size),
            patience=int(args.mlp_n_iter_no_change),
            val_fraction=float(args.mlp_validation_fraction),
            seed=int(seed),
            device=str(args.torch_device),
            num_threads=int(args.torch_num_threads),
        )
    raise ValueError("--model-type must be logistic, histgb, mlp, torch_mlp, or auto")


def quick_balance_metrics(g: np.ndarray, X_raw: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Fast balance summary used only for internal repair-validation selection."""
    g = np.asarray(g, dtype=int)
    groups = sorted(int(x) for x in np.unique(g))
    if len(groups) != 2:
        return {"mean_abs_smd": float("nan"), "max_abs_smd": float("nan"), "ess_ratio": effective_sample_size(w) / max(len(w), 1)}
    a, b = groups
    smds = []
    for j in range(X_raw.shape[1]):
        x = np.asarray(X_raw[:, j], dtype=float)
        ma, va = weighted_mean_var(x[g == a], w[g == a])
        mb, vb = weighted_mean_var(x[g == b], w[g == b])
        if not np.isfinite(ma) or not np.isfinite(mb):
            continue
        sd = math.sqrt(max(0.5 * (va + vb), 1e-12))
        smds.append(abs((ma - mb) / sd))
    if not smds:
        smds = [float("nan")]
    return {
        "mean_abs_smd": float(np.nanmean(smds)),
        "max_abs_smd": float(np.nanmax(smds)),
        "ess": effective_sample_size(w),
        "ess_ratio": float(effective_sample_size(w) / max(len(w), 1)),
        "weight_max": float(np.nanmax(w)),
        "weight_p99": float(np.nanquantile(w, 0.99)),
    }


def fit_one_propensity_candidate(
    model_type: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    args: argparse.Namespace,
    positive_group: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    model = build_propensity_estimator(model_type, args, seed)
    model.fit(X_train, y_train)
    e_val = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
    e_val = np.clip(e_val, float(args.propensity_clip), 1.0 - float(args.propensity_clip))
    w_val = make_weights(g_val, e_val, positive_group, args.weight_scheme, args.trim_quantile, args.max_weight)
    bal = quick_balance_metrics(g_val, X_val_raw, w_val)
    loss = (
        float(bal["mean_abs_smd"])
        + float(args.selection_max_smd_weight) * float(bal["max_abs_smd"])
        + float(args.selection_ess_weight) * max(0.0, 1.0 - float(bal["ess_ratio"]))
    )
    metrics = {
        "candidate_model_type": str(model_type),
        "prop_val_auc": safe_auc(y_val, e_val),
        "prop_val_log_loss": safe_log_loss(y_val, e_val),
        "prop_val_mean_abs_smd": bal["mean_abs_smd"],
        "prop_val_max_abs_smd": bal["max_abs_smd"],
        "prop_val_ess": bal["ess"],
        "prop_val_ess_ratio": bal["ess_ratio"],
        "prop_val_weight_p99": bal["weight_p99"],
        "prop_val_weight_max": bal["weight_max"],
        "selection_loss": float(loss),
    }
    return {"model": model, "scaler": scaler}, metrics


def fit_propensity_model(
    train: dict[str, np.ndarray],
    args: argparse.Namespace,
    positive_group: int,
) -> dict[str, Any]:
    """Fit deployment-safe e(H) on D_repair.

    The model is selected, when requested, by an internal repair-validation split.
    The selection criterion prioritizes weighted balance and effective sample size,
    not gender-prediction AUC alone.  After selection, the chosen model is refit on
    the full D_repair split.
    """
    X_raw, feature_names = probability_features(
        train["probs"], args.feature_mode, top_k=args.top_k_prob_features, split=train,
        score_profile_grid=args.score_profile_grid,
        simple_score_grid=args.simple_score_grid,
        saps_raps_score_grid=args.saps_raps_score_grid,
    )
    y = (np.asarray(train["g"], dtype=int) == int(positive_group)).astype(int)
    g = np.asarray(train["g"], dtype=int)

    if str(args.model_type).lower() == "auto":
        candidates = [x.strip().lower() for x in str(args.auto_candidates).split(",") if x.strip()]
    else:
        candidates = [str(args.model_type).lower()]
    candidates = [c for c in candidates if c in {"logistic", "histgb", "mlp", "torch_mlp"}]
    if not candidates:
        raise ValueError("No valid propensity model candidates. Use logistic,histgb,mlp,torch_mlp.")

    idx = np.arange(len(y))
    if float(args.repair_val_frac) > 0 and len(np.unique(y)) == 2:
        tr_idx, val_idx = train_test_split(
            idx,
            test_size=float(args.repair_val_frac),
            random_state=int(args.model_random_seed),
            stratify=y,
        )
    else:
        tr_idx, val_idx = idx, idx

    selection_rows = []
    for c_i, mt in enumerate(candidates):
        _, row = fit_one_propensity_candidate(
            mt,
            X_raw[tr_idx],
            y[tr_idx],
            X_raw[val_idx],
            y[val_idx],
            g[val_idx],
            args,
            positive_group,
            seed=int(args.model_random_seed) + 17 * c_i,
        )
        selection_rows.append(row)

    selection_df = pd.DataFrame(selection_rows).sort_values("selection_loss", ascending=True)
    selected_model_type = str(selection_df.iloc[0]["candidate_model_type"])

    # Refit selected model on all D_repair after model selection.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    model = build_propensity_estimator(selected_model_type, args, int(args.model_random_seed) + 1009)
    model.fit(X, y)
    e_train = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    e_train_clip = np.clip(e_train, float(args.propensity_clip), 1.0 - float(args.propensity_clip))
    w_train = make_weights(g, e_train_clip, positive_group, args.weight_scheme, args.trim_quantile, args.max_weight)
    final_bal = quick_balance_metrics(g, X_raw, w_train)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_mode": args.feature_mode,
        "model_type": selected_model_type,
        "requested_model_type": str(args.model_type),
        "auto_candidates": candidates,
        "selection_table": selection_df.to_dict(orient="records"),
        "repair_val_frac": float(args.repair_val_frac),
        "selection_metric": "mean_abs_smd + selection_max_smd_weight*max_abs_smd + selection_ess_weight*(1-ESS_ratio)",
        "top_k": int(args.top_k_prob_features),
        "score_profile_grid": str(args.score_profile_grid),
        "simple_score_grid": str(args.simple_score_grid),
        "saps_raps_score_grid": str(args.saps_raps_score_grid),
        "positive_group": int(positive_group),
        "repair_train_auc": safe_auc(y, e_train_clip),
        "repair_train_log_loss": safe_log_loss(y, e_train_clip),
        "repair_train_mean_abs_smd": final_bal["mean_abs_smd"],
        "repair_train_max_abs_smd": final_bal["max_abs_smd"],
        "repair_train_ess_ratio": final_bal["ess_ratio"],
    }


def predict_propensity(bundle: dict[str, Any], split: dict[str, np.ndarray], clip: float) -> tuple[np.ndarray, np.ndarray]:
    X_raw, names = probability_features(
        split["probs"], bundle["feature_mode"], top_k=int(bundle.get("top_k", 3)), split=split,
        score_profile_grid=bundle.get("score_profile_grid", DEFAULT_SCORE_PROFILE_GRID),
        simple_score_grid=bundle.get("simple_score_grid", DEFAULT_SIMPLE_SCORE_GRID),
        saps_raps_score_grid=bundle.get("saps_raps_score_grid", DEFAULT_SAPS_RAPS_SCORE_GRID),
    )
    expected = list(bundle.get("feature_names", names))
    if len(names) != len(expected):
        raise ValueError(
            f"Feature dimension mismatch: model expects {len(expected)} features but this split produced {len(names)}. "
            "Check feature_mode and whether length artifacts are consistently exported."
        )
    X = bundle["scaler"].transform(X_raw)
    e = np.asarray(bundle["model"].predict_proba(X)[:, 1], dtype=float)
    e = np.clip(e, float(clip), 1.0 - float(clip))
    return e, X_raw



def raw_propensity_weights(
    g: np.ndarray,
    e: np.ndarray,
    positive_group: int,
    scheme: str,
) -> np.ndarray:
    """Return untrimmed, unnormalized propensity weights.

    Keeping this separate lets us estimate the trimming cap once on D_repair and
    reuse that frozen cap on cal/test.  This avoids letting the held-out split
    determine its own post-processing cutoff.
    """
    g = np.asarray(g, dtype=int)
    e = np.asarray(e, dtype=float)
    pos = g == int(positive_group)
    pi_pos = float(pos.mean())
    pi_neg = 1.0 - pi_pos
    scheme = str(scheme).lower()
    if scheme == "ipw":
        w = np.where(pos, pi_pos / e, pi_neg / (1.0 - e))
    elif scheme == "overlap":
        w = np.where(pos, 1.0 - e, e)
    elif scheme == "matching":
        w = np.minimum(e, 1.0 - e) / np.where(pos, e, 1.0 - e)
    else:
        raise ValueError("--weight-scheme must be ipw, overlap, or matching")
    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = np.nan
    return w


def estimate_weight_trim_cap(
    g: np.ndarray,
    e: np.ndarray,
    positive_group: int,
    scheme: str,
    trim_quantile: float | None,
) -> float | None:
    """Estimate a raw-weight trimming cap on a designated development split."""
    if trim_quantile is None or float(trim_quantile) >= 1.0:
        return None
    w = raw_propensity_weights(g, e, positive_group, scheme)
    finite = w[np.isfinite(w)]
    if finite.size == 0:
        return None
    return float(np.nanquantile(finite, float(trim_quantile)))


def make_weights(
    g: np.ndarray,
    e: np.ndarray,
    positive_group: int,
    scheme: str,
    trim_quantile: float,
    max_weight: float | None,
    fixed_trim_cap: float | None = None,
) -> np.ndarray:
    w = raw_propensity_weights(g, e, positive_group, scheme)
    if fixed_trim_cap is not None and np.isfinite(float(fixed_trim_cap)):
        w = np.minimum(w, float(fixed_trim_cap))
    elif trim_quantile is not None and float(trim_quantile) < 1.0:
        cap = float(np.nanquantile(w, float(trim_quantile)))
        w = np.minimum(w, cap)
    if max_weight is not None and float(max_weight) > 0:
        w = np.minimum(w, float(max_weight))
    med = float(np.nanmedian(w[np.isfinite(w)])) if np.isfinite(w).any() else 1.0
    if med > 0:
        w = w / med
    return w


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import log_loss

        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        return float("nan")


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if len(w) == 0 or float(np.sum(w * w)) <= 0:
        return float("nan")
    return float((np.sum(w) ** 2) / np.sum(w * w))


# -----------------------------------------------------------------------------
# Balance diagnostics
# -----------------------------------------------------------------------------


def weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not mask.any() or float(w[mask].sum()) <= 0:
        return float("nan"), float("nan")
    x = x[mask]
    w = w[mask]
    m = float(np.sum(w * x) / np.sum(w))
    v = float(np.sum(w * (x - m) ** 2) / np.sum(w))
    return m, v


def feature_balance_table(
    split_name: str,
    g: np.ndarray,
    X_raw: np.ndarray,
    names: list[str],
    w: np.ndarray,
    meta: dict[str, Any],
) -> pd.DataFrame:
    groups = sorted(int(x) for x in np.unique(g))
    if len(groups) != 2:
        raise ValueError("Expected two groups for balance diagnostics.")
    a, b = groups
    rows = []
    for j, nm in enumerate(names):
        x = X_raw[:, j]
        ma0, va0 = weighted_mean_var(x[g == a], np.ones(int((g == a).sum())))
        mb0, vb0 = weighted_mean_var(x[g == b], np.ones(int((g == b).sum())))
        ma1, va1 = weighted_mean_var(x[g == a], w[g == a])
        mb1, vb1 = weighted_mean_var(x[g == b], w[g == b])
        sd0 = math.sqrt(max(0.5 * (va0 + vb0), 1e-12))
        sd1 = math.sqrt(max(0.5 * (va1 + vb1), 1e-12))
        rows.append({
            "split": split_name,
            "feature": nm,
            "group_a": a,
            "group_a_name": group_name(meta, a),
            "group_b": b,
            "group_b_name": group_name(meta, b),
            "mean_a_unweighted": ma0,
            "mean_b_unweighted": mb0,
            "smd_unweighted": float((ma0 - mb0) / sd0),
            "mean_a_weighted": ma1,
            "mean_b_weighted": mb1,
            "smd_weighted": float((ma1 - mb1) / sd1),
            "abs_smd_unweighted": float(abs((ma0 - mb0) / sd0)),
            "abs_smd_weighted": float(abs((ma1 - mb1) / sd1)),
        })
    return pd.DataFrame(rows)


def categorical_balance_table(
    split_name: str,
    g: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    meta: dict[str, Any],
    variable_name: str,
) -> pd.DataFrame:
    groups = sorted(int(x) for x in np.unique(g))
    cats = sorted(int(x) for x in np.unique(z))
    rows = []
    for cat in cats:
        row: dict[str, Any] = {
            "split": split_name,
            "variable": variable_name,
            "category": cat,
            "category_name": occupation_name(meta, cat) if variable_name == "true_occupation_posthoc" else str(cat),
        }
        props_un = []
        props_w = []
        for gr in groups:
            m = g == gr
            n_g = int(m.sum())
            p0 = float(np.mean(z[m] == cat)) if n_g > 0 else np.nan
            wg = w[m]
            zz = z[m]
            p1 = float(np.sum(wg * (zz == cat)) / np.sum(wg)) if np.sum(wg) > 0 else np.nan
            row[f"p_unweighted_group_{gr}"] = p0
            row[f"p_weighted_group_{gr}"] = p1
            row[f"group_{gr}_name"] = group_name(meta, gr)
            props_un.append(p0)
            props_w.append(p1)
        if len(props_un) == 2:
            row["abs_diff_unweighted"] = float(abs(props_un[0] - props_un[1]))
            row["abs_diff_weighted"] = float(abs(props_w[0] - props_w[1]))
        rows.append(row)
    return pd.DataFrame(rows)


def weight_summary_table(split_name: str, g: np.ndarray, e: np.ndarray, w: np.ndarray, meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    all_groups = [None] + sorted(int(x) for x in np.unique(g))
    for gr in all_groups:
        mask = np.ones(len(g), dtype=bool) if gr is None else (g == gr)
        ww = np.asarray(w[mask], dtype=float)
        ee = np.asarray(e[mask], dtype=float)
        rows.append({
            "split": split_name,
            "group": "all" if gr is None else int(gr),
            "group_name": "all" if gr is None else group_name(meta, int(gr)),
            "n": int(mask.sum()),
            "ess": effective_sample_size(ww),
            "ess_ratio": float(effective_sample_size(ww) / max(mask.sum(), 1)),
            "weight_mean": float(np.nanmean(ww)),
            "weight_sd": float(np.nanstd(ww)),
            "weight_min": float(np.nanmin(ww)),
            "weight_p50": float(np.nanquantile(ww, 0.50)),
            "weight_p90": float(np.nanquantile(ww, 0.90)),
            "weight_p95": float(np.nanquantile(ww, 0.95)),
            "weight_p99": float(np.nanquantile(ww, 0.99)),
            "weight_max": float(np.nanmax(ww)),
            "propensity_mean": float(np.nanmean(ee)),
            "propensity_min": float(np.nanmin(ee)),
            "propensity_p50": float(np.nanquantile(ee, 0.50)),
            "propensity_max": float(np.nanmax(ee)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Curve tables
# -----------------------------------------------------------------------------


def weighted_average(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not mask.any() or float(w[mask].sum()) <= 0:
        return float("nan")
    return float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))


def group_curve(
    split: dict[str, np.ndarray],
    split_name: str,
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    weights: np.ndarray,
    diagnostic_policies: list[str],
) -> pd.DataFrame:
    rows = []
    for policy in diagnostic_policies:
        for alpha in alphas:
            thr = thresholds[policy][float(alpha)]
            size = set_sizes(split["S"], split["g"], thr)
            cov = covered(split["S"], split["y"], split["g"], thr)
            for g in sorted(int(x) for x in np.unique(split["g"])):
                m = np.asarray(split["g"], dtype=int) == g
                wg = np.asarray(weights[m], dtype=float)
                rows.append({
                    "split": split_name,
                    "policy": policy,
                    "alpha": float(alpha),
                    "group": int(g),
                    "group_name": group_name(meta, g),
                    "n": int(m.sum()),
                    "weight_sum": float(np.sum(wg)),
                    "ess": effective_sample_size(wg),
                    "avg_set_size": weighted_average(size[m], wg),
                    "coverage": weighted_average(cov[m], wg),
                    "unweighted_avg_set_size": float(np.mean(size[m])) if m.any() else np.nan,
                    "unweighted_coverage": float(np.mean(cov[m])) if m.any() else np.nan,
                    "threshold": threshold_value(thr, int(g)),
                    "estimand": "learned_propensity_adjusted_group_curve",
                })
    return pd.DataFrame(rows)


def cell_curve(
    split: dict[str, np.ndarray],
    split_name: str,
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    meta: dict[str, Any],
    weights: np.ndarray,
    diagnostic_policies: list[str],
) -> pd.DataFrame:
    rows = []
    for policy in diagnostic_policies:
        for alpha in alphas:
            thr = thresholds[policy][float(alpha)]
            size = set_sizes(split["S"], split["g"], thr)
            cov = covered(split["S"], split["y"], split["g"], thr)
            for g in sorted(int(x) for x in np.unique(split["g"])):
                for z in sorted(int(x) for x in np.unique(split["z"])):
                    m = (np.asarray(split["g"], dtype=int) == g) & (np.asarray(split["z"], dtype=int) == z)
                    if not m.any():
                        continue
                    wz = np.asarray(weights[m], dtype=float)
                    rows.append({
                        "split": split_name,
                        "policy": policy,
                        "alpha": float(alpha),
                        "group": int(g),
                        "group_name": group_name(meta, g),
                        "audit": int(z),
                        "profession_id": profession_id(meta, z),
                        "occupation_name": occupation_name(meta, z),
                        "audit_name": occupation_name(meta, z),
                        "n": int(m.sum()),
                        "weight_sum": float(np.sum(wz)),
                        "ess": effective_sample_size(wz),
                        "avg_set_size": weighted_average(size[m], wz),
                        "coverage": weighted_average(cov[m], wz),
                        "unweighted_avg_set_size": float(np.mean(size[m])) if m.any() else np.nan,
                        "unweighted_coverage": float(np.mean(cov[m])) if m.any() else np.nan,
                        "threshold": threshold_value(thr, int(g)),
                        "audit_note": "true occupation is used only for post-hoc cell audit, not for propensity features",
                    })
    return pd.DataFrame(rows)


def aggregate_gap(curve: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(int(x) for x in curve["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    a, b = groups
    rows = []
    for (split, policy, alpha), sub in curve.groupby(["split", "policy", "alpha"], sort=True):
        sa = sub[sub.group == a].iloc[0]
        sb = sub[sub.group == b].iloc[0]
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "group_a": int(a),
            "group_a_name": str(sa.group_name),
            "group_b": int(b),
            "group_b_name": str(sb.group_name),
            "delta_V_a_minus_b": float(sa.avg_set_size - sb.avg_set_size),
            "delta_C_a_minus_b": float(sa.coverage - sb.coverage),
            "unweighted_delta_V_a_minus_b": float(sa.unweighted_avg_set_size - sb.unweighted_avg_set_size),
            "unweighted_delta_C_a_minus_b": float(sa.unweighted_coverage - sb.unweighted_coverage),
            "V_a": float(sa.avg_set_size),
            "V_b": float(sb.avg_set_size),
            "C_a": float(sa.coverage),
            "C_b": float(sb.coverage),
            "V_a_unweighted": float(sa.unweighted_avg_set_size),
            "V_b_unweighted": float(sb.unweighted_avg_set_size),
            "C_a_unweighted": float(sa.unweighted_coverage),
            "C_b_unweighted": float(sb.unweighted_coverage),
        })
    return pd.DataFrame(rows)


def simpson_gap_table(
    gap: pd.DataFrame,
    cell: pd.DataFrame,
    n_min: int,
    tau_agg: float,
    tau_loc: float,
    alpha0: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = sorted(int(x) for x in cell["group"].unique())
    a, b = groups
    gap_idx = gap.set_index(["split", "policy", "alpha"])
    rows = []
    for (split, policy, alpha, z), sub in cell.groupby(["split", "policy", "alpha", "audit"], sort=True):
        ga = sub[sub.group == a]
        gb = sub[sub.group == b]
        if ga.empty or gb.empty:
            continue
        ra = ga.iloc[0]
        rb = gb.iloc[0]
        d = gap_idx.loc[(split, policy, float(alpha))]
        agg = float(d.delta_V_a_minus_b)
        agg_raw = float(d.unweighted_delta_V_a_minus_b)
        loc = float(ra.avg_set_size - rb.avg_set_size)
        loc_raw = float(ra.unweighted_avg_set_size - rb.unweighted_avg_set_size)
        supported = bool(int(ra.n) >= int(n_min) and int(rb.n) >= int(n_min))
        reversal = bool(supported and abs(agg) >= tau_agg and abs(loc) >= tau_loc and agg * loc < 0)
        raw_reversal = bool(supported and abs(agg_raw) >= tau_agg and abs(loc_raw) >= tau_loc and agg_raw * loc_raw < 0)
        rows.append({
            "split": split,
            "policy": policy,
            "alpha": float(alpha),
            "audit": int(z),
            "profession_id": int(ra.profession_id),
            "occupation_name": str(ra.occupation_name),
            "audit_name": str(ra.occupation_name),
            "group_a": int(a),
            "group_a_name": str(ra.group_name),
            "group_b": int(b),
            "group_b_name": str(rb.group_name),
            "n_a_z": int(ra.n),
            "n_b_z": int(rb.n),
            "ess_a_z": float(ra.ess),
            "ess_b_z": float(rb.ess),
            "V_a_z": float(ra.avg_set_size),
            "V_b_z": float(rb.avg_set_size),
            "C_a_z": float(ra.coverage),
            "C_b_z": float(rb.coverage),
            "aggregate_gap_delta": agg,
            "local_gap_delta_z": loc,
            "unweighted_aggregate_gap_delta": agg_raw,
            "unweighted_local_gap_delta_z": loc_raw,
            "delta_times_delta_z": float(agg * loc),
            "reversal_strength": float(max(-agg * loc, 0.0)),
            "supported": supported,
            "is_simpson_reversal": reversal,
            "is_unweighted_simpson_reversal": raw_reversal,
        })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long, long
    summary_rows = []
    active = long[long.is_simpson_reversal].copy()
    for (split, policy, z), sub in long.groupby(["split", "policy", "audit"], sort=True):
        near = sub.iloc[int(np.argmin(np.abs(sub.alpha.to_numpy(float) - float(alpha0))))]
        summary_rows.append({
            "split": split,
            "policy": policy,
            "audit": int(z),
            "profession_id": int(sub.profession_id.iloc[0]),
            "occupation_name": str(sub.occupation_name.iloc[0]),
            "supported_all_alpha": bool(sub.supported.all()),
            "num_reversal_alpha": int(sub.is_simpson_reversal.sum()),
            "num_unweighted_reversal_alpha": int(sub.is_unweighted_simpson_reversal.sum()),
            "total_reversal_strength": float(sub.reversal_strength.sum()),
            "alpha0": float(alpha0),
            "is_reversal_at_alpha0_nearest": bool(near.is_simpson_reversal),
            "nearest_alpha": float(near.alpha),
            "aggregate_gap_at_alpha0_nearest": float(near.aggregate_gap_delta),
            "local_gap_at_alpha0_nearest": float(near.local_gap_delta_z),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["split", "policy", "num_reversal_alpha", "total_reversal_strength"], ascending=[True, True, False, False])
    return long, summary, active


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------



def alpha_mid_widths(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size <= 1:
        return np.full_like(alpha, 0.01)
    diffs = np.diff(alpha)
    left = np.r_[diffs[0], diffs]
    right = np.r_[diffs, diffs[-1]]
    return 0.5 * (left + right)


def shade_reversal_regions(ax: Any, alpha: np.ndarray, active: np.ndarray, color: str | None = None, alpha_level: float = 0.22) -> None:
    c = colors()
    color = color or c["shade"]
    alpha = np.asarray(alpha, dtype=float)
    active = np.asarray(active, dtype=bool)
    if alpha.size == 0 or not active.any():
        return
    widths = alpha_mid_widths(alpha)
    for a, w, is_active in zip(alpha, widths, active):
        if is_active:
            ax.axvspan(a - 0.5 * w, a + 0.5 * w, color=color, alpha=alpha_level, linewidth=0)


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


def reversal_cmap() -> Any:
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        "learned_adjusted_reversal",
        ["#F8FAFC", "#FEF3C7", "#FDBA74", "#FB7185", "#BE123C"],
    )


def clean_label(x: Any) -> str:
    return str(x).replace("_", " ")


def local_marker_for_group(group_label: str, idx: int) -> str:
    low = str(group_label).lower()
    if "male" in low and "female" not in low:
        return "v"
    if "female" in low:
        return "^"
    return ["v", "^", "s", "D"][idx % 4]


def order_occupations(summary: pd.DataFrame, cell_or_long: pd.DataFrame, mode: str) -> list[int]:
    audits = sorted(int(x) for x in cell_or_long.audit.unique()) if not cell_or_long.empty else []
    if mode == "label" or summary.empty:
        return audits
    sort_cols = [c for c in ["num_reversal_alpha", "total_reversal_strength"] if c in summary.columns]
    if sort_cols:
        s = summary.groupby("audit", as_index=False)[sort_cols].sum()
        s = s.sort_values(sort_cols + ["audit"], ascending=[False] * len(sort_cols) + [True])
        return [int(x) for x in s.audit]
    return audits


def plot_main(curve: pd.DataFrame, long: pd.DataFrame, summary: pd.DataFrame, score: str, split_name: str, out_dir: Path, formats: list[str], order_mode: str) -> list[str]:
    """Analyse-style adjusted diagnosis figure: adjusted only, no raw curves."""
    setup_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    c = colors()
    policy = "groupwise" if "groupwise" in set(curve.policy) else str(curve.policy.iloc[0])
    f = curve[(curve.split == split_name) & (curve.policy == policy)].copy()
    d = long[(long.split == split_name) & (long.policy == policy)].copy()
    if f.empty or d.empty:
        return []
    alpha_values = sorted(float(x) for x in d.alpha.unique())
    ssplit = summary[(summary.split == split_name) & (summary.policy == policy)] if not summary.empty else summary
    order = order_occupations(ssplit, d, order_mode)
    labels = {int(r.audit): str(r.occupation_name) for r in d[["audit", "occupation_name"]].drop_duplicates().itertuples(index=False)}
    mat = np.zeros((len(order), len(alpha_values)), dtype=float)
    active = np.zeros_like(mat, dtype=bool)
    for row in d.itertuples(index=False):
        i = order.index(int(row.audit))
        j = alpha_values.index(float(row.alpha))
        if bool(row.is_simpson_reversal):
            mat[i, j] = float(row.reversal_strength)
            active[i, j] = True

    fig = plt.figure(figsize=(18.0, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.04, 1.35], height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for idx, gr in enumerate(sorted(f.group.unique())):
        sub = f[f.group == gr].sort_values("alpha")
        nm = str(sub.group_name.iloc[0])
        ax_a.plot(sub.alpha, sub.avg_set_size, color=group_color(nm, idx), linestyle="-", linewidth=3.0, marker="o", markersize=5.0, label=nm)
    ax_a.set_title(f"A. Adjusted aggregate curve ({split_name}, {policy})", loc="left", fontweight="bold")
    ax_a.set_xlabel(r"miscoverage level $\alpha$")
    ax_a.set_ylabel("weighted average set size")
    ax_a.legend(title="group", loc="best", frameon=True, framealpha=0.94)

    vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.0
    im = ax_b.imshow(mat, aspect="auto", interpolation="none", cmap=reversal_cmap(), vmin=0.0, vmax=max(vmax, 1e-12))
    xticks, xticklabels = alpha_tick_positions(alpha_values, max_ticks=12)
    ax_b.set_xticks(xticks)
    ax_b.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels([clean_label(labels.get(z, z)) for z in order])
    ax_b.set_xlabel(r"miscoverage level $\alpha$")
    ax_b.set_ylabel("occupation")
    ax_b.set_title("B. Adjusted Simpson reversal evidence", loc="left", fontweight="bold")
    add_heatmap_grid(ax_b, len(order), len(alpha_values))
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.034, pad=0.018)
    cbar.set_label("adjusted reversal strength")

    if ssplit is not None and not ssplit.empty:
        counts = ssplit.set_index("audit").reindex(order).reset_index()
        vals = counts["num_reversal_alpha"].fillna(0).astype(float).to_numpy()
        strength = counts["total_reversal_strength"].fillna(0.0).astype(float).to_numpy()
    else:
        vals = active.sum(axis=1).astype(float)
        strength = mat.sum(axis=1)
    y = np.arange(len(order))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(float(np.nanmax(strength)), 1e-12))
    bar_cols = [reversal_cmap()(norm(st)) if v > 0 else "#CBD5E1" for v, st in zip(vals, strength)]
    ax_c.barh(y, vals, color=bar_cols, height=0.70, edgecolor="white", linewidth=0.9)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([clean_label(labels.get(z, z)) for z in order])
    ax_c.invert_yaxis()
    ax_c.set_xlabel(f"number of adjusted reversal grid levels (out of {len(alpha_values)})")
    ax_c.set_title("C. Reversal persistence by occupation", loc="left", fontweight="bold")
    max_val = max(float(np.nanmax(vals)), 1.0)
    for yi, val in zip(y, vals):
        ax_c.text(val + 0.16, yi, f"{int(val)} of {len(alpha_values)}", va="center", fontsize=9.6, color="#334155")
    ax_c.set_xlim(0, max_val + 3.0)

    active_d = d[d.is_simpson_reversal].copy()
    if not active_d.empty:
        chosen = active_d.iloc[int(active_d.reversal_strength.argmax())]
    else:
        chosen = d.iloc[int(d.reversal_strength.argmax())]
    audit = int(chosen.audit)
    label = clean_label(labels.get(audit, audit))
    local = d[d.audit == audit].sort_values("alpha").copy()
    alpha = local.alpha.to_numpy(dtype=float)
    agg_y = local.aggregate_gap_delta.to_numpy(dtype=float)
    loc_y = local.local_gap_delta_z.to_numpy(dtype=float)
    all_y = np.r_[agg_y[np.isfinite(agg_y)], loc_y[np.isfinite(loc_y)], [0.0]]
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = max(0.025, 0.12 * (ymax - ymin + 1e-12))
    shade_reversal_regions(ax_d, alpha, local.is_simpson_reversal.to_numpy(dtype=bool), color=c["shade"], alpha_level=0.27)
    gap_label = f"{str(chosen.group_a_name)} - {str(chosen.group_b_name)}"
    ax_d.plot(alpha, agg_y, color=c["dark"], linewidth=2.8, marker="o", markersize=4.5, label=f"aggregate gap ({gap_label})")
    ax_d.plot(alpha, loc_y, color=c["gold"], linewidth=2.4, linestyle="--", marker="D", markersize=4.0, label=f"{label} gap ({gap_label})")
    ax_d.axhline(0.0, linewidth=1.2, color=c["dark"])
    ax_d.set_ylim(ymin - pad, ymax + pad)
    ax_d.set_xlabel(r"miscoverage level $\alpha$")
    ax_d.set_ylabel("adjusted set-size gap")
    ax_d.set_title(f"D. Strongest adjusted witness: {label}", loc="left", fontweight="bold")
    ax_d.legend(loc="best", frameon=True, framealpha=0.94)

    fig.suptitle(f"BioBias learned-propensity adjusted curve diagnosis ({score.upper()}, {split_name})", y=0.992, fontsize=16.8, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, out_dir / f"fig_main_biobias_learned_propensity_curve_{score}_{split_name}", formats)


def plot_occupation_grid(cell: pd.DataFrame, curve: pd.DataFrame, long: pd.DataFrame, summary: pd.DataFrame, score: str, split_name: str, out_dir: Path, formats: list[str], order_mode: str) -> list[str]:
    """Adjusted occupation grid with pale-yellow Simpson regions; raw curves are intentionally omitted."""
    setup_plotting()
    import matplotlib.pyplot as plt

    c = colors()
    policy = "groupwise" if "groupwise" in set(cell.policy) else str(cell.policy.iloc[0])
    ccell = cell[(cell.split == split_name) & (cell.policy == policy)].copy()
    f = curve[(curve.split == split_name) & (curve.policy == policy)].copy()
    d = long[(long.split == split_name) & (long.policy == policy)].copy()
    if ccell.empty or f.empty:
        return []
    ssplit = summary[(summary.split == split_name) & (summary.policy == policy)] if not summary.empty else summary
    audits = order_occupations(ssplit, ccell, order_mode)
    n = len(audits)
    ncols = 5 if n >= 5 else max(1, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.65 * nrows), sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    group_ids = sorted(int(x) for x in ccell.group.unique())
    y_values = ccell.avg_set_size.astype(float).tolist() + f.avg_set_size.astype(float).tolist()
    finite_y = np.asarray([v for v in y_values if np.isfinite(v)], dtype=float)
    ylim = None
    if finite_y.size:
        lo, hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        pad = max(0.10, 0.10 * (hi - lo + 1e-12))
        ylim = (max(0.0, lo - pad), hi + pad)
    summary_counts = ssplit.set_index("audit")["num_reversal_alpha"].to_dict() if ssplit is not None and not ssplit.empty else {}
    for ax, z in zip(axes, audits):
        subz = ccell[ccell.audit == z].copy()
        sub_gap = d[d.audit == z].sort_values("alpha")
        title = str(subz.occupation_name.iloc[0]) if not subz.empty else f"occupation {z}"
        if not sub_gap.empty:
            shade_reversal_regions(ax, sub_gap.alpha.to_numpy(float), sub_gap.is_simpson_reversal.to_numpy(dtype=bool), color=c["shade"], alpha_level=0.22)
        for idx, gr in enumerate(group_ids):
            agg = f[f.group == gr].sort_values("alpha")
            if not agg.empty:
                nm = str(agg.group_name.iloc[0])
                ax.plot(agg.alpha, agg.avg_set_size, color=group_color(nm, idx), linestyle=":", linewidth=2.1, alpha=0.82, label=f"{nm} aggregate" if z == audits[0] else None)
            loc = subz[subz.group == gr].sort_values("alpha")
            if loc.empty:
                continue
            nm = str(loc.group_name.iloc[0])
            col = group_color(nm, idx)
            ax.plot(loc.alpha, loc.avg_set_size, color=col, linestyle="-", linewidth=2.15, marker=local_marker_for_group(nm, idx), markersize=4.3, markeredgecolor="white", markeredgewidth=0.55, label=f"{nm} local adjusted" if z == audits[0] else None)
        reversals = int(summary_counts.get(z, int(sub_gap.is_simpson_reversal.sum()) if not sub_gap.empty else 0))
        denom = int(len(sub_gap)) if not sub_gap.empty else 0
        ax.set_title(f"{clean_label(title)}\n{reversals}/{denom} adjusted reversals", fontsize=10.0, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("weighted set size")
    for ax in axes[len(audits):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels() if len(axes) else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{score.upper()} learned-propensity adjusted occupation curve grid ({split_name})", y=1.065, fontsize=15.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, out_dir / f"fig_appendix_learned_propensity_occupation_curve_grid_{score}_{split_name}", formats)


def plot_gap_heatmap(long: pd.DataFrame, summary: pd.DataFrame, score: str, split_name: str, out_dir: Path, formats: list[str], order_mode: str) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    policy = "groupwise" if "groupwise" in set(long.policy) else str(long.policy.iloc[0])
    d = long[(long.split == split_name) & (long.policy == policy)].copy()
    if d.empty:
        return []
    audits = order_occupations(summary[(summary.split == split_name) & (summary.policy == policy)] if not summary.empty else summary, d, order_mode)
    alphas = sorted(float(x) for x in d.alpha.unique())
    mat = np.full((len(audits), len(alphas)), np.nan)
    rev = np.zeros((len(audits), len(alphas)), dtype=bool)
    name = {}
    for _, r in d.iterrows():
        i = audits.index(int(r.audit))
        j = alphas.index(float(r.alpha))
        mat[i, j] = float(r.local_gap_delta_z)
        rev[i, j] = bool(r.is_simpson_reversal)
        name[int(r.audit)] = str(r.occupation_name)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(max(9.5, 0.55 * len(alphas)), max(4.8, 0.42 * len(audits))))
    cmap = mcolors.LinearSegmentedColormap.from_list("gap", ["#2563EB", "#F8FAFC", "#E11D48"])
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap, vmin=-vmax, vmax=vmax)
    xticks, xticklabels = alpha_tick_positions(alphas, max_ticks=14)
    ax.set_yticks(range(len(audits)))
    ax.set_yticklabels([clean_label(name.get(z, f"occupation {z}")) for z in audits])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlabel(r"miscoverage level $\alpha$")
    ax.set_title(f"{score.upper()} learned-adjusted local gap heatmap; black dots = adjusted Simpson reversals ({split_name})")
    add_heatmap_grid(ax, len(audits), len(alphas))
    for i in range(len(audits)):
        for j in range(len(alphas)):
            if rev[i, j]:
                ax.plot(j, i, marker="o", color="#0F172A", markersize=4.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r"local adjusted gap $V_{a,z}^w - V_{b,z}^w$")
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_appendix_learned_propensity_gap_curve_grid_{score}_{split_name}", formats)

def plot_weight_distribution(weight_df: pd.DataFrame, split_name: str, out_dir: Path, formats: list[str]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    d = weight_df[weight_df.split == split_name].copy()
    if d.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    axes[0].hist(d["propensity"].to_numpy(float), bins=40)
    axes[0].set_title(f"Propensity distribution ({split_name})")
    axes[0].set_xlabel(r"$\hat e(H)$")
    axes[0].set_ylabel("count")
    axes[1].hist(d["weight"].to_numpy(float), bins=40)
    axes[1].set_title(f"Weight distribution ({split_name})")
    axes[1].set_xlabel("weight")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_weight_distribution_{split_name}", formats)


def plot_balance(balance: pd.DataFrame, split_name: str, out_dir: Path, formats: list[str], top_k: int = 20) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt

    d = balance[balance.split == split_name].copy()
    if d.empty:
        return []
    d["improvement"] = d["abs_smd_unweighted"] - d["abs_smd_weighted"]
    d = d.sort_values("abs_smd_unweighted", ascending=False).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.8, max(4.8, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.plot(d["abs_smd_unweighted"], y, marker="o", linestyle="", label="before")
    ax.plot(d["abs_smd_weighted"], y, marker="o", linestyle="", label="after")
    for yi, x0, x1 in zip(y, d["abs_smd_unweighted"], d["abs_smd_weighted"]):
        ax.plot([x0, x1], [yi, yi], color="#94A3B8", linewidth=1.2)
    ax.axvline(0.1, color="#E11D48", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"].astype(str))
    ax.set_xlabel("absolute standardized mean difference")
    ax.set_title(f"Feature balance before/after learned propensity weighting ({split_name})")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, out_dir / f"fig_balance_diagnostics_{split_name}", formats)


# -----------------------------------------------------------------------------
# Running a score/split
# -----------------------------------------------------------------------------


def add_scores(data: dict[str, dict[str, np.ndarray]], score: str, args: argparse.Namespace) -> None:
    for split_name, split in data.items():
        split["S"] = scores_from_probs(
            split["probs"], score, args.saps_lambda, args.raps_lambda, args.raps_k_reg,
            args.score_randomize, args.score_random_seed, args.score_temperature,
        )



def _plot_saved_tables_for_split(
    curve_path: str,
    cell_path: str,
    long_path: str,
    summary_path: str,
    score: str,
    split_name: str,
    figs_dir: str,
    formats: list[str],
    order_mode: str,
    queue: Any | None = None,
) -> list[str] | None:
    """Read saved curve tables and render one split's adjusted diagnostics.

    This function is intentionally top-level so it can be executed in a short-lived
    child process.  Isolating matplotlib rendering keeps cal/test runs stable when
    the parent process still has model/probability arrays in memory.
    """
    try:
        curve_plot = pd.read_csv(curve_path)
        cell_plot = pd.read_csv(cell_path)
        long_plot = pd.read_csv(long_path)
        summary_plot = pd.read_csv(summary_path)
        figs = Path(figs_dir)
        paths: list[str] = []
        paths += plot_main(curve_plot, long_plot, summary_plot, score, split_name, figs, formats, order_mode)
        paths += plot_occupation_grid(cell_plot, curve_plot, long_plot, summary_plot, score, split_name, figs, formats, order_mode)
        paths += plot_gap_heatmap(long_plot, summary_plot, score, split_name, figs, formats, order_mode)
        if queue is not None:
            queue.put({"ok": True, "paths": paths})
            return None
        return paths
    except Exception as exc:  # pragma: no cover - defensive plotting path
        if queue is not None:
            queue.put({"ok": False, "error": repr(exc)})
            return None
        raise


def _plot_saved_tables_for_split_isolated(
    curve_path: Path,
    cell_path: Path,
    long_path: Path,
    summary_path: Path,
    score: str,
    split_name: str,
    figs: Path,
    formats: list[str],
    order_mode: str,
    timeout_seconds: float,
) -> list[str]:
    """Render one split in a short-lived subprocess, with inline fallback."""
    try:
        import multiprocessing as mp
        method = "spawn" if "spawn" in mp.get_all_start_methods() else "fork"
        ctx = mp.get_context(method)
        q = ctx.Queue()
        proc = ctx.Process(
            target=_plot_saved_tables_for_split,
            args=(str(curve_path), str(cell_path), str(long_path), str(summary_path), score, split_name, str(figs), formats, order_mode, q),
        )
        proc.start()
        proc.join(float(timeout_seconds))
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(2)
            raise TimeoutError(f"plotting split={split_name} exceeded {timeout_seconds} seconds")
        if proc.exitcode != 0:
            raise RuntimeError(f"plotting subprocess exited with code {proc.exitcode}")
        if q.empty():
            raise RuntimeError("plotting subprocess returned no result")
        res = q.get()
        if not res.get("ok", False):
            raise RuntimeError(str(res.get("error", "unknown plotting error")))
        return list(res.get("paths", []))
    except Exception as exc:
        print(f"    isolated plotting failed ({exc}); falling back to inline plotting", flush=True)
        return list(_plot_saved_tables_for_split(str(curve_path), str(cell_path), str(long_path), str(summary_path), score, split_name, str(figs), formats, order_mode, None) or [])


def run_curve_for_score(
    score: str,
    eval_splits: dict[str, dict[str, np.ndarray]],
    threshold_source: dict[str, np.ndarray],
    weight_by_split: dict[str, np.ndarray],
    meta: dict[str, Any],
    args: argparse.Namespace,
    score_root: Path,
) -> dict[str, Any]:
    add_scores({**eval_splits, "__threshold__": threshold_source}, score, args)
    thresholds = build_thresholds(threshold_source, args.alpha_grid_values)
    policies = ["pooled", "groupwise"] if args.diagnostic_policy == "both" else [args.diagnostic_policy]

    curve_parts = []
    cell_parts = []
    for split_name, split in eval_splits.items():
        w = weight_by_split[split_name]
        curve_parts.append(group_curve(split, split_name, thresholds, args.alpha_grid_values, meta, w, policies))
        cell_parts.append(cell_curve(split, split_name, thresholds, args.alpha_grid_values, meta, w, policies))
    curve = pd.concat(curve_parts, ignore_index=True)
    cell = pd.concat(cell_parts, ignore_index=True)
    gap = aggregate_gap(curve)
    long, summary, active = simpson_gap_table(gap, cell, args.n_min, args.tau_agg, args.tau_loc, args.alpha0)

    # The raw probability/score arrays are no longer needed after the tabular
    # curves have been computed.  Clearing them before matplotlib rendering
    # prevents memory pressure when cal/test use a high-dimensional safe_score_profile.
    try:
        seen_ids = set()
        for _split in list(eval_splits.values()) + [threshold_source]:
            if id(_split) not in seen_ids:
                seen_ids.add(id(_split))
                _split.clear()
        import gc
        gc.collect()
    except Exception:
        pass

    tables = mkdir(score_root / "tables")
    figs = mkdir(score_root / "figures")
    curve_path = tables / "learned_adjusted_group_curve.csv"
    cell_path = tables / "learned_adjusted_cell_curve.csv"
    gap_path = tables / "learned_adjusted_gap_curve.csv"
    long_path = tables / "learned_adjusted_simpson_gap_curve_long.csv"
    summary_path = tables / "learned_adjusted_simpson_curve_summary.csv"
    active_path = tables / "active_learned_adjusted_simpson_curve_cells.csv"
    curve.to_csv(curve_path, index=False)
    cell.to_csv(cell_path, index=False)
    gap.to_csv(gap_path, index=False)
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    active.to_csv(active_path, index=False)

    n_rows_curve = int(len(curve))
    n_active_reversals = int(len(active))
    split_names_for_plots = sorted(curve["split"].astype(str).unique().tolist())

    # Plot from CSV-backed tables rather than computation-backed frames.  This is
    # slightly more I/O, but it makes cal/test plotting robust for large runs and
    # keeps the plotting product exactly reproducible from saved tables.
    try:
        import gc
        del curve, cell, gap, long, summary, active
        gc.collect()
    except Exception:
        pass
    fig_paths: list[str] = []
    if not bool(getattr(args, "no_curve_figures", False)):
        curve_plot = pd.read_csv(curve_path)
        cell_plot = pd.read_csv(cell_path)
        long_plot = pd.read_csv(long_path)
        summary_plot = pd.read_csv(summary_path)
        for split_name in split_names_for_plots:
            print(f"  plotting adjusted diagnostics for split={split_name} ...", flush=True)
            fig_paths += plot_main(curve_plot, long_plot, summary_plot, score, split_name, figs, args.figure_formats_list, args.occupation_order)
            fig_paths += plot_occupation_grid(cell_plot, curve_plot, long_plot, summary_plot, score, split_name, figs, args.figure_formats_list, args.occupation_order)
            fig_paths += plot_gap_heatmap(long_plot, summary_plot, score, split_name, figs, args.figure_formats_list, args.occupation_order)
            print("    done plotting split", flush=True)
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    return {
        "score": score,
        "score_root": str(score_root),
        "n_rows_curve": n_rows_curve,
        "n_active_reversals": n_active_reversals,
        "figures": fig_paths,
    }


def make_weight_trace(
    split_name: str,
    split: dict[str, np.ndarray],
    e: np.ndarray,
    w: np.ndarray,
    meta: dict[str, Any],
    positive_group: int,
) -> pd.DataFrame:
    pred = np.argmax(split["probs"], axis=1)
    rows = {
        "split": split_name,
        "row": np.arange(len(split["g"])),
        "group": split["g"].astype(int),
        "group_name": [group_name(meta, int(x)) for x in split["g"]],
        "positive_group": int(positive_group),
        "positive_group_name": group_name(meta, int(positive_group)),
        "propensity": e,
        "weight": w,
        "predicted_occupation": pred.astype(int),
        "predicted_occupation_name": [occupation_name(meta, int(x)) for x in pred],
        "true_occupation_for_audit_only": split["z"].astype(int),
        "true_occupation_name_for_audit_only": [occupation_name(meta, int(x)) for x in split["z"]],
    }
    for optional in ["sample_id", "source_split", "source_index", "artifact_index"]:
        if optional in split:
            rows[optional] = split[optional]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train deployment-safe learned propensity adjustment on BioBias D_repair and audit adjusted curves.")
    p.add_argument("--seed-dir", type=Path, default=None, help="Path to outputs_biobias_train/<run>/seed_X")
    p.add_argument("--run-root", type=Path, default=None, help="Path to outputs_biobias_train/<run>; use with --seed")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prob-dir", type=Path, default=None, help="Path directly containing probability artifacts")
    p.add_argument("--output-root", type=Path, default=None, help="Default: <seed_dir>/curve_adjusted_learned_propensity")
    p.add_argument("--clean-output", action="store_true")

    p.add_argument("--feature-mode", choices=["safe_main", "safe_main_len", "safe_topk", "safe_probs", "safe_probs_len", "safe_score_profile", "safe_score_profile_len"], default="safe_main")
    p.add_argument("--model-type", choices=["logistic", "histgb", "mlp", "torch_mlp", "auto"], default="logistic",
                   help="auto selects among --auto-candidates using an internal D_repair validation split and a balance/ESS criterion.")
    p.add_argument("--auto-candidates", default="logistic,torch_mlp")
    p.add_argument("--positive-group-name", default="Male")
    p.add_argument("--weight-scheme", choices=["ipw", "overlap", "matching"], default="overlap")
    p.add_argument("--propensity-clip", type=float, default=0.02)
    p.add_argument("--trim-quantile", type=float, default=0.995)
    p.add_argument("--max-weight", type=float, default=None)
    p.add_argument("--top-k-prob-features", type=int, default=3)
    p.add_argument("--score-profile-grid", default=DEFAULT_SCORE_PROFILE_GRID,
                   help="Mass levels used by safe_score_profile cumulative-mass fingerprints. Does not use labels or calibration quantiles.")
    p.add_argument("--simple-score-grid", default=DEFAULT_SIMPLE_SCORE_GRID,
                   help="Fixed SIMPLE score cutoffs for label-free candidate-count fingerprints.")
    p.add_argument("--saps-raps-score-grid", default=DEFAULT_SAPS_RAPS_SCORE_GRID,
                   help="Fixed SAPS/RAPS score cutoffs for label-free candidate-count fingerprints.")
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--histgb-max-iter", type=int, default=250)
    p.add_argument("--histgb-learning-rate", type=float, default=0.04)
    p.add_argument("--histgb-l2", type=float, default=0.03)
    p.add_argument("--histgb-max-leaf-nodes", type=int, default=15)
    p.add_argument("--mlp-hidden-layers", default="64,32")
    p.add_argument("--mlp-alpha", type=float, default=1e-3, help="L2 regularization for sklearn MLPClassifier.")
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-max-iter", type=int, default=250, help="Only used by sklearn --model-type mlp.")
    p.add_argument("--torch-mlp-epochs", type=int, default=40, help="Only used by --model-type torch_mlp.")
    p.add_argument("--torch-mlp-batch-size", type=int, default=4096)
    p.add_argument("--torch-device", default="auto", help="auto, cpu, or cuda for torch_mlp.")
    p.add_argument("--torch-num-threads", type=int, default=1, help="PyTorch CPU threads for torch_mlp; 1 is often faster for these small tabular models.")
    p.add_argument("--mlp-validation-fraction", type=float, default=0.15)
    p.add_argument("--mlp-n-iter-no-change", type=int, default=12)
    p.add_argument("--disable-mlp-early-stopping", action="store_true")
    p.add_argument("--repair-val-frac", type=float, default=0.25,
                   help="Internal validation split inside D_repair for propensity-model selection. Does not touch D_cal/D_test.")
    p.add_argument("--selection-max-smd-weight", type=float, default=0.25)
    p.add_argument("--selection-ess-weight", type=float, default=0.50)
    p.add_argument("--model-random-seed", type=int, default=0)

    p.add_argument("--score", choices=["simple", "aps", "saps", "raps", "all"], default="all")
    p.add_argument("--saps-lambda", type=float, default=0.2)
    p.add_argument("--raps-lambda", type=float, default=0.2)
    p.add_argument("--raps-k-reg", type=int, default=5)
    p.add_argument("--score-temperature", type=float, default=1.0)
    p.add_argument("--score-randomize", action="store_true")
    p.add_argument("--score-random-seed", type=int, default=0)

    p.add_argument("--alpha0", type=float, default=0.10)
    p.add_argument("--alpha-grid", default="dense_safe")
    p.add_argument("--diagnostic-policy", choices=["pooled", "groupwise", "both"], default="groupwise")
    p.add_argument("--n-min", type=int, default=40)
    p.add_argument("--tau-agg", type=float, default=0.005)
    p.add_argument("--tau-loc", type=float, default=0.02)
    p.add_argument("--occupation-order", choices=["evidence", "label"], default="evidence")
    p.add_argument("--figure-formats", default="pdf,png,svg")
    p.add_argument("--plot-timeout-seconds", type=float, default=180.0,
                   help="Reserved for external/isolated plotting; kept for command compatibility.")
    p.add_argument("--no-curve-figures", action="store_true",
                   help="Write curve tables but skip score-specific curve figures. Use biobias_plot_adjusted_curve_tables.py to render from saved tables.")
    p.add_argument("--write-sample-tables", action="store_true", help="Write row-level propensity and weight tables. May be large.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.mlp_early_stopping = not bool(args.disable_mlp_early_stopping)
    args.alpha_grid_values = parse_grid(args.alpha_grid)
    if float(args.alpha0) not in args.alpha_grid_values:
        args.alpha_grid_values = sorted(args.alpha_grid_values + [float(args.alpha0)])
    args.figure_formats_list = [x.strip().lower() for x in args.figure_formats.split(",") if x.strip()]

    seed_dir = resolve_seed_dir(args)
    prob_dir = resolve_prob_dir(args, seed_dir)
    if args.output_root is None:
        out_root = seed_dir / DEFAULT_OUTPUT_DIRNAME
    else:
        out_root = args.output_root.expanduser().resolve()
        if out_root.name.startswith("seed_"):
            out_root = out_root / DEFAULT_OUTPUT_DIRNAME
    if args.clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root = unique_output_path(out_root)
    mkdir(out_root)
    tables_root = mkdir(out_root / "tables")
    figs_root = mkdir(out_root / "figures")

    meta = load_meta(prob_dir, seed_dir)
    data = load_splits(prob_dir, ["repair"])
    repair = data["repair"]
    pos_group = positive_group_id(meta, repair["g"], args.positive_group_name)

    print(f"Training deployment-safe propensity model on D_repair: e(H)=P(G={group_name(meta, pos_group)}|H)")
    print(f"Feature mode: {args.feature_mode}; true occupation is NOT used in H.")
    bundle = fit_propensity_model(
        repair,
        args=args,
        positive_group=pos_group,
    )
    bundle["meta_summary"] = {
        "positive_group": int(pos_group),
        "positive_group_name": group_name(meta, pos_group),
        "feature_safety": "deployment_safe: features are constructed from frozen model probabilities only; true occupation is excluded",
    }
    save_pickle(bundle, out_root / "learned_propensity_model.pkl")
    jdump({k: v for k, v in bundle.items() if k not in {"model", "scaler"}}, out_root / "learned_propensity_model_manifest.json")
    pd.DataFrame(bundle.get("selection_table", [])).to_csv(tables_root / "propensity_model_selection_on_repair_validation.csv", index=False)

    e, X_raw = predict_propensity(bundle, repair, args.propensity_clip)
    w = make_weights(repair["g"], e, pos_group, args.weight_scheme, args.trim_quantile, args.max_weight)
    weight_by_split = {"repair": w}

    weight_summary = weight_summary_table("repair", repair["g"], e, w, meta)
    weight_summary.to_csv(tables_root / "learned_propensity_weight_summary.csv", index=False)
    feature_balance = feature_balance_table("repair", repair["g"], X_raw, bundle["feature_names"], w, meta)
    feature_balance.to_csv(tables_root / "balance_diagnostics_safe_features.csv", index=False)
    true_occ_balance = categorical_balance_table("repair", repair["g"], repair["z"], w, meta, "true_occupation_posthoc")
    true_occ_balance.to_csv(tables_root / "balance_diagnostics_true_occupation_posthoc.csv", index=False)

    trace = make_weight_trace("repair", repair, e, w, meta, pos_group)
    if args.write_sample_tables:
        trace.to_csv(tables_root / "learned_propensity_weights_repair_sample_trace.csv", index=False)
    else:
        trace.head(5000).to_csv(tables_root / "learned_propensity_weights_repair_sample_trace_head5000.csv", index=False)

    plot_weight_distribution(trace, "repair", figs_root, args.figure_formats_list)
    plot_balance(feature_balance, "repair", figs_root, args.figure_formats_list)

    scores = ["simple", "aps", "saps", "raps"] if args.score == "all" else [args.score]
    manifests = []
    for sc in scores:
        print(f"Repair-split audit for score={sc} ...")
        score_root = mkdir(out_root / sc)
        # Use a fresh load so different score computations do not overwrite threshold source aliases unexpectedly.
        score_data = load_splits(prob_dir, ["repair"])
        man = run_curve_for_score(sc, score_data, score_data["repair"], weight_by_split, meta, args, score_root)
        manifests.append(man)
        print(f"  saved {score_root}; active learned-adjusted reversals: {man['n_active_reversals']}")

    jdump({
        "script": "biobias_adjusted_curve_learned_propensity.py",
        "probability_dir": str(prob_dir),
        "seed_dir": str(seed_dir),
        "output_root": str(out_root),
        "training_split_for_propensity": "repair",
        "threshold_source_split_for_diagnostic": "repair",
        "evaluation_splits": ["repair"],
        "positive_group": int(pos_group),
        "positive_group_name": group_name(meta, pos_group),
        "feature_mode": args.feature_mode,
        "score_profile_grid": args.score_profile_grid,
        "simple_score_grid": args.simple_score_grid,
        "saps_raps_score_grid": args.saps_raps_score_grid,
        "feature_safety": "deployment-safe: no true occupation/label in H; true occupation is used only for post-hoc audit grouping",
        "requested_model_type": args.model_type,
        "selected_model_type": bundle.get("model_type"),
        "auto_candidates": bundle.get("auto_candidates"),
        "repair_val_frac": args.repair_val_frac,
        "selection_metric": bundle.get("selection_metric"),
        "weight_scheme": args.weight_scheme,
        "propensity_clip": args.propensity_clip,
        "trim_quantile": args.trim_quantile,
        "max_weight": args.max_weight,
        "repair_train_auc": bundle.get("repair_train_auc"),
        "repair_train_log_loss": bundle.get("repair_train_log_loss"),
        "repair_train_mean_abs_smd": bundle.get("repair_train_mean_abs_smd"),
        "repair_train_max_abs_smd": bundle.get("repair_train_max_abs_smd"),
        "repair_train_ess_ratio": bundle.get("repair_train_ess_ratio"),
        "scores": manifests,
    }, out_root / "analysis_manifest.json")
    print(f"Done. Output root: {out_root}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
