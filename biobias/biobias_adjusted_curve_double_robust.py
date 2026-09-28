#!/usr/bin/env python3
"""
BioBias double-robust standardized curve pipeline.

Validity boundary
-----------------
The conformal score S, the calibration quantile q_hat_alpha, and the prediction
set C_alpha(X) are fixed before any adjustment is performed.  Propensity and
outcome models are used only after C_alpha(X) has been formed, to estimate and
plot standardized set-size curves.  In particular, this script does not use
weighted conformal quantiles and does not modify prediction-set membership.

Outputs
-------
Default output root:

    <seed_dir>/curve_adjusted_double_robust/

Inside it, four parallel estimator folders are created:

    raw_curve/
    ipw_overlap_curve/
    or_curve/
    aipw_dr_curve/

Each estimator folder contains score subfolders (simple/saps/raps) with
analyse-style tables and figures.

Updated DR design
-----------------
The main IPW overlap curve may keep using the richer H from --feature-mode.
The OR and AIPW-DR curves can instead use a separate common conservative
L_DR via --dr-feature-mode balanced_l_dr.  In that mode the outcome model is
forbidden from seeing full probability vectors, score-profile/set-size
fingerprints, calibration thresholds, true labels, or true occupations.  AIPW
then uses e_L(L_DR) and m_g,alpha(L_DR), preserving the common-covariate
structure needed for a strict AIPW interpretation.

Pooled outcome designs
----------------------
Use --outcome-design pooled_additive for group intercepts plus a common L_DR
difficulty model, or --outcome-design pooled_group_aware for one pooled
Q(A,L_DR) model with group indicators.  These are weaker than separate
group-specific OR models and are useful when the goal is to show that AIPW
residual correction improves over a deliberately low-complexity OR.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Shared BioBias I/O, score, plotting, and propensity utilities.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from biobias_curve_utils import (  # noqa: E402
    DEFAULT_SCORE_PROFILE_GRID,
    DEFAULT_SIMPLE_SCORE_GRID,
    DEFAULT_SAPS_RAPS_SCORE_GRID,
    add_scores,
    aggregate_gap,
    build_thresholds,
    categorical_balance_table,
    cell_curve,
    covered,
    effective_sample_size,
    feature_balance_table,
    fit_propensity_model,
    group_curve,
    group_name,
    jdump,
    load_meta,
    load_pickle,
    load_splits,
    make_weight_trace,
    make_weights,
    estimate_weight_trim_cap,
    mkdir,
    occupation_name,
    parse_grid,
    path_has_content,
    plot_balance,
    plot_gap_heatmap,
    plot_main,
    plot_occupation_grid,
    plot_weight_distribution,
    positive_group_id,
    predict_propensity,
    profession_id,
    qhat,
    resolve_prob_dir,
    resolve_seed_dir,
    save_pickle,
    set_sizes,
    threshold_value,
    true_scores,
    unique_output_path,
    weight_summary_table,
    weighted_average,
)

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scikit-learn. Install with: pip install scikit-learn") from exc

DR_OUTPUT_DIRNAME = "curve_adjusted_double_robust"
ESTIMATOR_DIRS = ["raw_curve", "ipw_overlap_curve", "or_curve", "aipw_dr_curve"]

# Feature modes that are too close to a set-size oracle for the OR/AIPW outcome
# model when --forbid-outcome-setsize-proxies is enabled.  They remain allowed
# for the main IPW balancing model because IPW does not regress on set size.
OUTCOME_PROXY_RISK_FEATURE_MODES = {
    "safe_probs",
    "safe_probs_len",
    "safe_score_profile",
    "safe_score_profile_len",
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def parse_csv_list(text: str, allowed: list[str] | None = None) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if allowed is not None:
        bad = [x for x in out if x not in allowed]
        if bad:
            raise ValueError(f"Unexpected value(s) {bad}; allowed={allowed}")
    return out


def subset_split(split: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    out: dict[str, np.ndarray] = {}
    for k, v in split.items():
        arr = np.asarray(v)
        if arr.shape[0] == mask.shape[0]:
            out[k] = arr[mask]
        else:
            out[k] = v
    return out


def stable_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lo), hi)


def target_tilt(e: np.ndarray, mode: str, clip: float) -> np.ndarray:
    e = stable_clip(e, clip, 1.0 - clip)
    if mode == "overlap":
        r = e * (1.0 - e)
    elif mode == "uniform":
        r = np.ones_like(e, dtype=float)
    else:
        raise ValueError(f"Unsupported --target-tilt {mode!r}")
    return np.asarray(r, dtype=float)


def group_pi(e: np.ndarray, group: int, positive_group: int, clip: float) -> np.ndarray:
    e = stable_clip(e, clip, 1.0 - clip)
    return e if int(group) == int(positive_group) else (1.0 - e)


def enforce_nonincreasing_rows(mat: np.ndarray) -> np.ndarray:
    """Project row-wise predictions to a non-increasing curve over alpha."""
    arr = np.asarray(mat, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return arr
    for j in range(1, arr.shape[1]):
        arr[:, j] = np.minimum(arr[:, j], arr[:, j - 1])
    return arr


@dataclass
class ConstantRegressor:
    mean_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.tile(np.asarray(self.mean_, dtype=float)[None, :], (X.shape[0], 1))


@dataclass
class AdditiveOffsetRegressor:
    """A shared difficulty model plus a group-specific intercept.

    This is used by --outcome-design pooled_additive.  The shared base_model is
    fitted to residuals after subtracting group means, and the offset_ is the
    group mean vector.  Because group identity is not passed into the base_model,
    OR cannot learn group-by-L interactions; AIPW must recover remaining
    discrepancies through the residual correction term.
    """

    base_model: Any
    offset_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.base_model.predict(np.asarray(X, dtype=float))
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred + np.asarray(self.offset_, dtype=float)[None, :]


def _augment_with_group_indicator(X: np.ndarray, g: np.ndarray | int, groups: list[int]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if np.isscalar(g):
        g_arr = np.full(X.shape[0], int(g), dtype=int)
    else:
        g_arr = np.asarray(g, dtype=int)
        if g_arr.shape[0] != X.shape[0]:
            raise ValueError("group array length must match X")
    mapping = {int(v): j for j, v in enumerate(groups)}
    Z = np.zeros((X.shape[0], len(groups)), dtype=float)
    for i, gv in enumerate(g_arr):
        if int(gv) in mapping:
            Z[i, mapping[int(gv)]] = 1.0
    return np.column_stack([X, Z])


@dataclass
class GroupAwareRegressor:
    """One pooled Q(A,L) model evaluated at a fixed group value.

    This is used by --outcome-design pooled_group_aware.  Unlike the pure
    pooled_additive design, this can learn limited group-by-L interactions, but
    because there is only one pooled model its complexity is lower than fitting
    separate group-specific outcome regressions.  It is useful when we want OR
    to remain imperfect while allowing AIPW residual correction to improve it.
    """

    base_model: Any
    group: int
    groups: list[int]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = _augment_with_group_indicator(np.asarray(X, dtype=float), int(self.group), list(self.groups))
        pred = self.base_model.predict(X_aug)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred



# -----------------------------------------------------------------------------
# Conservative common L_DR features and DR propensity
# -----------------------------------------------------------------------------


def _one_hot(idx: np.ndarray, k: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    idx = np.asarray(idx, dtype=int)
    out = np.zeros((idx.shape[0], k), dtype=float)
    ok = (idx >= 0) & (idx < k)
    out[np.arange(idx.shape[0])[ok], idx[ok]] = 1.0
    return out, [f"{prefix}_{j}" for j in range(k)]


def _fixed_bin_one_hot(x: np.ndarray, bins: list[float], prefix: str) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x, dtype=float)
    bins_arr = np.asarray(bins, dtype=float)
    # np.digitize creates len(bins)+1 bins.  Fixed bins are allowed because they
    # do not depend on D_cal quantiles, alpha, true labels, or realized set sizes.
    b = np.digitize(x, bins_arr, right=False)
    m = len(bins_arr) + 1
    out = np.zeros((x.shape[0], m), dtype=float)
    out[np.arange(x.shape[0]), np.clip(b, 0, m - 1)] = 1.0
    names = []
    edges = [-np.inf] + list(bins_arr) + [np.inf]
    for j in range(m):
        names.append(f"{prefix}_bin_{j}_{edges[j]:.3g}_{edges[j+1]:.3g}")
    return out, names


def make_balanced_l_dr_features(split: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    """Construct a conservative common adjustment variable L_DR.

    L_DR is deliberately weaker than full safe_score_profile.  It uses only
    low-dimensional prediction-difficulty summaries from frozen probabilities
    plus coarse top-label identity.  It excludes full probability vectors, full
    sorted probability vectors, cumulative-mass curves, score-count fingerprints,
    calibration-threshold features, alpha-specific set-size proxies, true labels,
    and true occupations.
    """
    if "probs" not in split:
        raise KeyError("split is missing 'probs'; cannot build balanced_l_dr features")
    p = np.asarray(split["probs"], dtype=float)
    if p.ndim != 2:
        raise ValueError("split['probs'] must be a 2D array")
    n, k = p.shape
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)

    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1] if k >= 2 else np.zeros(n)
    p3 = sorted_p[:, 2] if k >= 3 else np.zeros(n)
    p4 = sorted_p[:, 3] if k >= 4 else np.zeros(n)
    p5 = sorted_p[:, 4] if k >= 5 else np.zeros(n)
    entropy = -np.sum(p * np.log(p), axis=1)
    entropy_norm = entropy / max(math.log(k), eps)
    effective_num = np.exp(entropy)
    l2_conc = np.sum(p * p, axis=1)
    margin12 = p1 - p2
    margin23 = p2 - p3
    top3_mass = sorted_p[:, : min(3, k)].sum(axis=1)
    top5_mass = sorted_p[:, : min(5, k)].sum(axis=1)

    blocks: list[np.ndarray] = []
    names: list[str] = []

    top_label_k = max(1, min(int(args.dr_top_label_k), k))
    for rnk in range(top_label_k):
        B, bn = _one_hot(order[:, rnk], k, f"ldr_top{rnk+1}_label")
        blocks.append(B)
        names += bn

    cont = np.column_stack([
        p1, p2, p3, p4, p5,
        entropy_norm, effective_num, l2_conc,
        margin12, margin23, top3_mass, top5_mass,
    ])
    blocks.append(cont)
    names += [
        "ldr_p_top1", "ldr_p_top2", "ldr_p_top3", "ldr_p_top4", "ldr_p_top5",
        "ldr_entropy_norm", "ldr_effective_num_labels", "ldr_prob_l2_concentration",
        "ldr_margin12", "ldr_margin23", "ldr_top3_mass", "ldr_top5_mass",
    ]

    if bool(args.dr_include_coarse_bins):
        for vals, bins, prefix in [
            (p1, [0.35, 0.50, 0.65, 0.80, 0.90], "ldr_p_top1"),
            (entropy_norm, [0.25, 0.40, 0.55, 0.70, 0.85], "ldr_entropy_norm"),
            (margin12, [0.05, 0.10, 0.20, 0.35, 0.55], "ldr_margin12"),
            (top5_mass, [0.60, 0.75, 0.85, 0.93, 0.97], "ldr_top5_mass"),
        ]:
            B, bn = _fixed_bin_one_hot(vals, bins, prefix)
            blocks.append(B)
            names += bn

    X = np.column_stack(blocks).astype(float)
    X[~np.isfinite(X)] = 0.0
    return X, names


def make_dr_features(
    split: dict[str, np.ndarray],
    args: argparse.Namespace,
    ipw_X: np.ndarray | None = None,
    ipw_feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    mode = str(args.dr_feature_mode).lower()
    if mode in {"same_as_ipw", "same_as_propensity", "same"}:
        if ipw_X is None:
            raise ValueError("same_as_ipw requested but ipw_X was not supplied")
        names = list(ipw_feature_names or [f"ipw_feature_{j}" for j in range(np.asarray(ipw_X).shape[1])])
        return np.asarray(ipw_X, dtype=float), names
    if mode in {"balanced_l_dr", "lean_difficulty", "balanced_score_profile"}:
        return make_balanced_l_dr_features(split, args)
    raise ValueError(f"Unsupported --dr-feature-mode {args.dr_feature_mode!r}")


def validate_dr_feature_safety(args: argparse.Namespace) -> None:
    if not bool(args.forbid_outcome_setsize_proxies):
        return
    mode = str(args.dr_feature_mode).lower()
    if mode in {"same_as_ipw", "same_as_propensity", "same"}:
        ipw_mode = str(args.feature_mode).lower()
        if ipw_mode in OUTCOME_PROXY_RISK_FEATURE_MODES:
            raise ValueError(
                "--forbid-outcome-setsize-proxies is enabled, but --dr-feature-mode "
                f"{args.dr_feature_mode!r} would expose OR/AIPW to IPW feature_mode={args.feature_mode!r}. "
                "Use --dr-feature-mode balanced_l_dr, or explicitly disable the safeguard."
            )


def fit_feature_propensity_model(
    X: np.ndarray,
    g: np.ndarray,
    positive_group: int,
    feature_names: list[str],
    args: argparse.Namespace,
    seed: int,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Fit e_L(L)=P(G=positive|L) for the strict common-L AIPW estimator."""
    X = np.asarray(X, dtype=float)
    y = (np.asarray(g, dtype=int) == int(positive_group)).astype(int)
    mt = str(model_type or args.dr_model_type)
    if mt == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(args.dr_logistic_c), max_iter=int(args.dr_logistic_max_iter), solver="lbfgs"),
        )
    elif mt == "histgb":
        model = HistGradientBoostingClassifier(
            max_iter=int(args.dr_histgb_max_iter),
            learning_rate=float(args.dr_histgb_learning_rate),
            l2_regularization=float(args.dr_histgb_l2),
            max_leaf_nodes=int(args.dr_histgb_max_leaf_nodes),
            random_state=seed,
        )
    elif mt == "mlp":
        hidden = tuple(int(x) for x in str(args.dr_mlp_hidden_layers).split(",") if x.strip())
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=float(args.dr_mlp_alpha),
                learning_rate_init=float(args.dr_mlp_lr),
                max_iter=int(args.dr_mlp_max_iter),
                early_stopping=True,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"Unsupported --dr-model-type {mt!r}")
    model.fit(X, y)
    return {
        "model": model,
        "model_type": mt,
        "feature_mode": args.dr_feature_mode,
        "feature_names": list(feature_names),
        "positive_group": int(positive_group),
        "n_features": int(X.shape[1]),
        "n_train": int(X.shape[0]),
        "note": "strict common-L DR propensity; used only for AIPW/OR target tilt, not for CP calibration",
    }


def predict_feature_propensity(bundle: dict[str, Any], X: np.ndarray, clip: float) -> np.ndarray:
    model = bundle["model"]
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        e = model.predict_proba(X)[:, 1]
    else:
        pred = model.predict(X)
        e = np.asarray(pred, dtype=float)
    return stable_clip(e, clip, 1.0 - clip)


def fit_regressor(X: np.ndarray, Y: np.ndarray, model_type: str, args: argparse.Namespace, seed: int) -> Any:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] == 0:
        return ConstantRegressor(np.zeros(Y.shape[1] if Y.ndim == 2 else 1, dtype=float))
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] < int(args.min_outcome_n):
        return ConstantRegressor(np.nanmean(Y, axis=0))
    model_type = str(model_type)
    if model_type == "constant":
        return ConstantRegressor(np.nanmean(Y, axis=0))
    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(args.outcome_ridge_alpha), random_state=seed))
    elif model_type == "histgb":
        base = HistGradientBoostingRegressor(
            max_iter=int(args.outcome_histgb_max_iter),
            learning_rate=float(args.outcome_histgb_learning_rate),
            l2_regularization=float(args.outcome_histgb_l2),
            max_leaf_nodes=int(args.outcome_histgb_max_leaf_nodes),
            random_state=seed,
        )
        model = MultiOutputRegressor(base)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=int(args.outcome_rf_estimators),
            min_samples_leaf=int(args.outcome_rf_min_samples_leaf),
            max_depth=None if int(args.outcome_rf_max_depth) <= 0 else int(args.outcome_rf_max_depth),
            random_state=seed,
            n_jobs=int(args.outcome_n_jobs),
        )
    elif model_type == "mlp":
        hidden = tuple(int(x) for x in str(args.outcome_mlp_hidden_layers).split(",") if x.strip())
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden,
                alpha=float(args.outcome_mlp_alpha),
                learning_rate_init=float(args.outcome_mlp_lr),
                max_iter=int(args.outcome_mlp_max_iter),
                early_stopping=True,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"Unsupported --outcome-model {model_type!r}")
    model.fit(X, Y)
    return model


def predict_regressor(model: Any, X: np.ndarray) -> np.ndarray:
    pred = model.predict(np.asarray(X, dtype=float))
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred


# -----------------------------------------------------------------------------
# Outcome regression nuisances for M_alpha=|C_alpha(X)| and coverage
# -----------------------------------------------------------------------------


def outcome_targets(
    split: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    policies: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for policy in policies:
        sizes = []
        covs = []
        for alpha in alphas:
            thr = thresholds[policy][float(alpha)]
            sizes.append(set_sizes(split["S"], split["g"], thr))
            covs.append(covered(split["S"], split["y"], split["g"], thr))
        out[policy] = {
            "size": np.column_stack(sizes).astype(float),
            "coverage": np.column_stack(covs).astype(float),
        }
    return out


def fit_pooled_additive_models(
    X_repair: np.ndarray,
    Y: np.ndarray,
    g_arr: np.ndarray,
    groups: list[int],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Fit m_{g,alpha}(L)=mu_{g,alpha}+f_alpha(L) with no group-L interaction.

    The common difficulty model f_alpha(L) is trained on pooled residuals
    Y - mean(Y | group).  Each returned group model adds its group intercept
    back to the common prediction.  This gives a low-complexity OR working
    model that captures common prediction difficulty but cannot fit separate
    male/female difficulty-response curves.
    """
    X_repair = np.asarray(X_repair, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    g_arr = np.asarray(g_arr, dtype=int)
    global_mean = np.nanmean(Y, axis=0)
    if not np.all(np.isfinite(global_mean)):
        global_mean = np.zeros(Y.shape[1], dtype=float)

    group_means: dict[int, np.ndarray] = {}
    Y_resid = np.zeros_like(Y, dtype=float)
    shrink = float(getattr(args, "outcome_additive_intercept_shrinkage", 0.0))
    shrink = min(max(shrink, 0.0), 1.0)
    for g in groups:
        m = g_arr == int(g)
        if np.any(m):
            g_mean = np.nanmean(Y[m], axis=0)
        else:
            g_mean = global_mean.copy()
        if not np.all(np.isfinite(g_mean)):
            g_mean = global_mean.copy()
        # Optional shrinkage toward the pooled mean.  Default 0 keeps ordinary
        # group intercepts; positive values make OR more conservative.
        g_mean = (1.0 - shrink) * g_mean + shrink * global_mean
        group_means[int(g)] = g_mean
        if np.any(m):
            Y_resid[m] = Y[m] - g_mean[None, :]

    # The common residual model receives only L_DR, never group identity.  Thus
    # tree/boosting outcome models cannot create group-specific split paths.
    base_model = fit_regressor(X_repair, Y_resid, args.outcome_model, args, seed)
    models = {int(g): AdditiveOffsetRegressor(base_model=base_model, offset_=group_means[int(g)]) for g in groups}
    info = {
        "additive_global_mean": global_mean.tolist(),
        "additive_group_means": {str(int(g)): group_means[int(g)].tolist() for g in groups},
        "additive_intercept_shrinkage": float(shrink),
        "additive_note": "pooled additive OR: group intercept + common L_DR difficulty model; no group-by-L interactions",
    }
    return models, info


def fit_pooled_group_aware_models(
    X_repair: np.ndarray,
    Y: np.ndarray,
    g_arr: np.ndarray,
    groups: list[int],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Fit one pooled Q(A,L) model and evaluate it at each group.

    The learner receives L_DR plus a group indicator.  With a weak learner such
    as a shallow RF with large leaves, this allows only limited group-by-L
    interactions and is less oracle-like than separate group-specific outcome
    models, while avoiding the too-flat behavior of a pure additive group
    intercept model.
    """
    X_repair = np.asarray(X_repair, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    g_arr = np.asarray(g_arr, dtype=int)
    X_aug = _augment_with_group_indicator(X_repair, g_arr, groups)
    base_model = fit_regressor(X_aug, Y, args.outcome_model, args, seed)
    models = {int(g): GroupAwareRegressor(base_model=base_model, group=int(g), groups=[int(v) for v in groups]) for g in groups}
    info = {
        "group_indicator_columns": [f"group_indicator_{int(g)}" for g in groups],
        "pooled_group_aware_note": "one pooled Q(A,L) model fitted on L_DR plus group indicators; use weak learners to limit group-by-L interactions",
    }
    return models, info



def fit_outcome_bundle(
    repair: dict[str, np.ndarray],
    X_repair: np.ndarray,
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    policies: list[str],
    groups: list[int],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    targets = outcome_targets(repair, thresholds, alphas, policies)
    n_alpha = len(alphas)
    max_size = int(np.asarray(repair["S"]).shape[1])
    bundle: dict[str, Any] = {
        "model_type": args.outcome_model,
        "outcome_design": args.outcome_design,
        "alphas": [float(a) for a in alphas],
        "policies": list(policies),
        "groups": [int(g) for g in groups],
        "n_alpha": int(n_alpha),
        "max_set_size": int(max_size),
        "models": {},
        "design_info": {},
    }
    g_arr = np.asarray(repair["g"], dtype=int)
    for policy in policies:
        Y = np.column_stack([targets[policy]["size"], targets[policy]["coverage"]])
        bundle["models"][policy] = {}
        if str(args.outcome_design) == "pooled_additive":
            models, info = fit_pooled_additive_models(
                X_repair=X_repair,
                Y=Y,
                g_arr=g_arr,
                groups=groups,
                args=args,
                seed=seed + 7919,
            )
            bundle["models"][policy] = models
            bundle["design_info"][policy] = info
        elif str(args.outcome_design) == "pooled_group_aware":
            models, info = fit_pooled_group_aware_models(
                X_repair=X_repair,
                Y=Y,
                g_arr=g_arr,
                groups=groups,
                args=args,
                seed=seed + 8929,
            )
            bundle["models"][policy] = models
            bundle["design_info"][policy] = info
        else:
            for g in groups:
                m = g_arr == int(g)
                if not np.any(m):
                    model = ConstantRegressor(np.nanmean(Y, axis=0))
                else:
                    model = fit_regressor(X_repair[m], Y[m], args.outcome_model, args, seed + 1009 * int(g))
                bundle["models"][policy][int(g)] = model
    return bundle



def predict_outcomes(
    bundle: dict[str, Any],
    X: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    n_alpha = int(bundle["n_alpha"])
    max_size = float(bundle["max_set_size"])
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for policy, models in bundle["models"].items():
        out[policy] = {}
        for g, model in models.items():
            pred = predict_regressor(model, X)
            if pred.shape[1] < 2 * n_alpha:
                raise RuntimeError(f"Outcome prediction has {pred.shape[1]} columns; expected {2 * n_alpha}.")
            size = pred[:, :n_alpha]
            cov = pred[:, n_alpha:2 * n_alpha]
            size = stable_clip(size, 0.0, max_size)
            cov = stable_clip(cov, 0.0, 1.0)
            if bool(args.outcome_monotone_project):
                size = enforce_nonincreasing_rows(size)
                cov = enforce_nonincreasing_rows(cov)
            out[policy][int(g)] = {"size": size, "coverage": cov}
    return out


# -----------------------------------------------------------------------------
# OR/AIPW curve tables
# -----------------------------------------------------------------------------


def dr_group_cell_curves(
    split: dict[str, np.ndarray],
    split_name: str,
    thresholds: dict[str, dict[float, float | dict[int, float]]],
    alphas: list[float],
    policies: list[str],
    meta: dict[str, Any],
    e: np.ndarray,
    r: np.ndarray,
    positive_group: int,
    outcome_pred: dict[str, dict[int, dict[str, np.ndarray]]],
    estimator: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate and occupation-cell curves for OR or AIPW."""
    if estimator not in {"or_curve", "aipw_dr_curve"}:
        raise ValueError(estimator)
    g_arr = np.asarray(split["g"], dtype=int)
    z_arr = np.asarray(split["z"], dtype=int)
    groups = sorted(int(x) for x in np.unique(g_arr))
    audits = sorted(int(x) for x in np.unique(z_arr))
    e = stable_clip(np.asarray(e, dtype=float), args.propensity_clip, 1.0 - args.propensity_clip)
    r = np.asarray(r, dtype=float)
    denom_all = float(np.sum(r))
    if denom_all <= 0:
        denom_all = float(len(r))
        r = np.ones_like(r, dtype=float)

    rows_group: list[dict[str, Any]] = []
    rows_cell: list[dict[str, Any]] = []
    for policy in policies:
        # Actual M_alpha and coverage under the fixed CP set.
        targets = outcome_targets(split, thresholds, alphas, [policy])[policy]
        size_actual = targets["size"]
        cov_actual = targets["coverage"]
        for j, alpha in enumerate(alphas):
            thr = thresholds[policy][float(alpha)]
            for g in groups:
                gmask = g_arr == int(g)
                pi = group_pi(e, int(g), positive_group, args.propensity_clip)
                m_size = np.asarray(outcome_pred[policy][int(g)]["size"][:, j], dtype=float)
                m_cov = np.asarray(outcome_pred[policy][int(g)]["coverage"][:, j], dtype=float)
                if estimator == "or_curve":
                    avg_size = float(np.sum(r * m_size) / denom_all)
                    avg_cov = float(np.sum(r * m_cov) / denom_all)
                else:
                    corr_size = np.zeros_like(m_size, dtype=float)
                    corr_cov = np.zeros_like(m_cov, dtype=float)
                    corr_size[gmask] = size_actual[gmask, j] - m_size[gmask]
                    corr_cov[gmask] = cov_actual[gmask, j] - m_cov[gmask]
                    avg_size = float((np.sum(r * m_size) + np.sum(r[gmask] / pi[gmask] * corr_size[gmask])) / denom_all)
                    avg_cov = float((np.sum(r * m_cov) + np.sum(r[gmask] / pi[gmask] * corr_cov[gmask])) / denom_all)
                rows_group.append({
                    "split": split_name,
                    "policy": policy,
                    "alpha": float(alpha),
                    "group": int(g),
                    "group_name": group_name(meta, g),
                    "n": int(gmask.sum()),
                    "weight_sum": denom_all,
                    "ess": effective_sample_size(r),
                    "avg_set_size": avg_size,
                    "coverage": avg_cov,
                    "unweighted_avg_set_size": float(np.mean(size_actual[gmask, j])) if np.any(gmask) else np.nan,
                    "unweighted_coverage": float(np.mean(cov_actual[gmask, j])) if np.any(gmask) else np.nan,
                    "threshold": threshold_value(thr, int(g)),
                    "estimator": estimator,
                    "estimand": estimator,
                    "target_tilt": args.target_tilt,
                    "threshold_source": "repair" if split_name == "repair" else "cal",
                    "validity_note": "curve estimator only; CP score/quantile/set membership unchanged",
                })
                for z in audits:
                    zmask = z_arr == int(z)
                    gzmask = gmask & zmask
                    denom_z = float(np.sum(r[zmask]))
                    if denom_z <= 0:
                        continue
                    if estimator == "or_curve":
                        avg_size_z = float(np.sum(r[zmask] * m_size[zmask]) / denom_z)
                        avg_cov_z = float(np.sum(r[zmask] * m_cov[zmask]) / denom_z)
                    else:
                        avg_size_z = float((
                            np.sum(r[zmask] * m_size[zmask])
                            + np.sum(r[gzmask] / pi[gzmask] * (size_actual[gzmask, j] - m_size[gzmask]))
                        ) / denom_z)
                        avg_cov_z = float((
                            np.sum(r[zmask] * m_cov[zmask])
                            + np.sum(r[gzmask] / pi[gzmask] * (cov_actual[gzmask, j] - m_cov[gzmask]))
                        ) / denom_z)
                    rows_cell.append({
                        "split": split_name,
                        "policy": policy,
                        "alpha": float(alpha),
                        "group": int(g),
                        "group_name": group_name(meta, g),
                        "audit": int(z),
                        "profession_id": profession_id(meta, z),
                        "occupation_name": occupation_name(meta, z),
                        "audit_name": occupation_name(meta, z),
                        "n": int(gzmask.sum()),
                        "weight_sum": denom_z,
                        "ess": effective_sample_size(r[zmask]),
                        "avg_set_size": avg_size_z,
                        "coverage": avg_cov_z,
                        "unweighted_avg_set_size": float(np.mean(size_actual[gzmask, j])) if np.any(gzmask) else np.nan,
                        "unweighted_coverage": float(np.mean(cov_actual[gzmask, j])) if np.any(gzmask) else np.nan,
                        "threshold": threshold_value(thr, int(g)),
                        "estimator": estimator,
                        "target_tilt": args.target_tilt,
                        "audit_note": "true occupation is used only after prediction sets are formed, for post-hoc cell audit",
                    })
    return pd.DataFrame(rows_group), pd.DataFrame(rows_cell)


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------


def canonical_and_alias_tables(
    tables_dir: Path,
    estimator: str,
    curve: pd.DataFrame,
    cell: pd.DataFrame,
    gap: pd.DataFrame,
    long: pd.DataFrame,
    summary: pd.DataFrame,
    active: pd.DataFrame,
) -> None:
    mkdir(tables_dir)
    canonical = {
        "aggregate_curve.csv": curve,
        "aggregate_gap_curve.csv": gap,
        "cell_curve.csv": cell,
        "simpson_gap_curve_long.csv": long,
        "simpson_curve_summary.csv": summary,
        "active_simpson_curve_cells.csv": active,
    }
    for name, df in canonical.items():
        df.to_csv(tables_dir / name, index=False)

    # Aliases consumed by the shared plotting functions.
    prefix = {
        "raw_curve": "raw",
        "ipw_overlap_curve": "learned_adjusted",
        "or_curve": "or",
        "aipw_dr_curve": "aipw_dr",
    }.get(estimator, estimator)
    alias = {
        f"{prefix}_group_curve.csv": curve,
        f"{prefix}_cell_curve.csv": cell,
        f"{prefix}_gap_curve.csv": gap,
        f"{prefix}_simpson_gap_curve_long.csv": long,
        f"{prefix}_simpson_curve_summary.csv": summary,
        f"active_{prefix}_simpson_curve_cells.csv": active,
    }
    for name, df in alias.items():
        df.to_csv(tables_dir / name, index=False)


def make_figure_aliases(figures_dir: Path, score: str, split_name: str, formats: list[str]) -> None:
    mapping = {
        f"fig_main_biobias_learned_propensity_curve_{score}_{split_name}": f"fig_main_biobias_curve_simpson_{score}_{split_name}",
        f"fig_appendix_learned_propensity_occupation_curve_grid_{score}_{split_name}": f"fig_appendix_occupation_curve_grid_{score}_{split_name}",
        f"fig_appendix_learned_propensity_gap_curve_grid_{score}_{split_name}": f"fig_appendix_gap_curve_grid_{score}_{split_name}",
    }
    for src_stem, dst_stem in mapping.items():
        for ext in formats:
            src = figures_dir / f"{src_stem}.{ext}"
            dst = figures_dir / f"{dst_stem}.{ext}"
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass


def save_and_plot_estimator_score(
    estimator: str,
    score: str,
    out_score_root: Path,
    curve: pd.DataFrame,
    cell: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gap = aggregate_gap(curve)
    long, summary, active = __import__("biobias_adjusted_curve_learned_propensity").simpson_gap_table(
        gap, cell, args.n_min, args.tau_agg, args.tau_loc, args.alpha0
    )
    # Annotate tables for easier downstream comparisons.
    for df in (curve, cell, gap, long, summary, active):
        if df is not None and not df.empty:
            df["estimator"] = estimator
    tables_dir = mkdir(out_score_root / "tables")
    figs_dir = mkdir(out_score_root / "figures")
    canonical_and_alias_tables(tables_dir, estimator, curve, cell, gap, long, summary, active)

    fig_paths: list[str] = []
    if not bool(args.no_curve_figures):
        splits = sorted(curve["split"].astype(str).unique().tolist())
        # Plot from the canonical CSVs so figures are exactly reproducible from saved tables.
        f_plot = pd.read_csv(tables_dir / "aggregate_curve.csv")
        c_plot = pd.read_csv(tables_dir / "cell_curve.csv")
        l_plot = pd.read_csv(tables_dir / "simpson_gap_curve_long.csv")
        s_plot = pd.read_csv(tables_dir / "simpson_curve_summary.csv")
        for split_name in splits:
            print(f"    plotting {estimator}/{score}/{split_name} ...", flush=True)
            fig_paths += plot_main(f_plot, l_plot, s_plot, score, split_name, figs_dir, args.figure_formats_list, args.occupation_order)
            fig_paths += plot_occupation_grid(c_plot, f_plot, l_plot, s_plot, score, split_name, figs_dir, args.figure_formats_list, args.occupation_order)
            fig_paths += plot_gap_heatmap(l_plot, s_plot, score, split_name, figs_dir, args.figure_formats_list, args.occupation_order)
            make_figure_aliases(figs_dir, score, split_name, args.figure_formats_list)
    return {
        "estimator": estimator,
        "score": score,
        "score_root": str(out_score_root),
        "n_curve_rows": int(len(curve)),
        "n_cell_rows": int(len(cell)),
        "n_active_reversals": int(len(active)),
        "figures": fig_paths,
    }


def summarize_outputs(out_root: Path, estimators: list[str], scores: list[str], diagnostic_policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for est in estimators:
        for sc in scores:
            p = out_root / est / sc / "tables" / "simpson_curve_summary.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if df.empty:
                continue
            policies = sorted(df["policy"].astype(str).unique())
            for split in sorted(df["split"].astype(str).unique()):
                for pol in policies:
                    if diagnostic_policy != "both" and pol != diagnostic_policy:
                        continue
                    sub = df[(df["split"].astype(str) == split) & (df["policy"].astype(str) == pol)]
                    if sub.empty:
                        continue
                    rows.append({
                        "estimator": est,
                        "score": sc,
                        "split": split,
                        "policy": pol,
                        "num_occupations_with_any_reversal": int((sub["num_reversal_alpha"] > 0).sum()),
                        "total_reversal_alpha_cells": int(sub["num_reversal_alpha"].sum()),
                        "total_unweighted_reversal_alpha_cells": int(sub["num_unweighted_reversal_alpha"].sum()),
                        "total_reversal_strength": float(sub["total_reversal_strength"].sum()),
                    })
    comp = pd.DataFrame(rows)
    red_rows = []
    if not comp.empty and "raw_curve" in set(comp["estimator"]):
        raw = comp[comp["estimator"] == "raw_curve"].copy()
        for row in comp.itertuples(index=False):
            base = raw[(raw.score == row.score) & (raw.split == row.split) & (raw.policy == row.policy)]
            if base.empty:
                continue
            base_row = base.iloc[0]
            red_rows.append({
                "estimator": row.estimator,
                "score": row.score,
                "split": row.split,
                "policy": row.policy,
                "raw_total_reversal_alpha_cells": int(base_row.total_reversal_alpha_cells),
                "estimator_total_reversal_alpha_cells": int(row.total_reversal_alpha_cells),
                "reduction_alpha_cells": int(base_row.total_reversal_alpha_cells - row.total_reversal_alpha_cells),
                "raw_total_reversal_strength": float(base_row.total_reversal_strength),
                "estimator_total_reversal_strength": float(row.total_reversal_strength),
                "reduction_strength": float(base_row.total_reversal_strength - row.total_reversal_strength),
            })
    return comp, pd.DataFrame(red_rows)


# -----------------------------------------------------------------------------
# Main score pipeline
# -----------------------------------------------------------------------------


def policies_from_args(args: argparse.Namespace) -> list[str]:
    return ["pooled", "groupwise"] if args.diagnostic_policy == "both" else [args.diagnostic_policy]


def run_score_pipeline(
    score: str,
    prob_dir: Path,
    meta: dict[str, Any],
    X_by_split: dict[str, np.ndarray],
    e_by_split: dict[str, np.ndarray],
    w_by_split: dict[str, np.ndarray],
    X_dr_by_split: dict[str, np.ndarray],
    e_dr_by_split: dict[str, np.ndarray],
    pos_group: int,
    out_root: Path,
    estimators: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    policies = policies_from_args(args)
    # Fresh load for this score because score arrays are stored in split dicts.
    data = load_splits(prob_dir, ["repair", "cal", "test"])
    add_scores(data, score, args)
    alphas = args.alpha_grid_values
    groups = sorted(int(x) for x in np.unique(data["repair"]["g"]))

    # Repair-threshold evaluation uses D_repair.
    thresholds_repair = build_thresholds(data["repair"], alphas)
    # Calibration-threshold evaluation uses unweighted D_cal.
    thresholds_cal = build_thresholds(data["cal"], alphas)

    need_outcome_nuisance = any(e in {"or_curve", "aipw_dr_curve"} for e in estimators)
    if need_outcome_nuisance:
        # Outcome nuisances are score/policy/threshold-source dependent because the
        # target M_alpha=|C_alpha(X)| depends on the conformal threshold.
        outcome_repair = fit_outcome_bundle(
            data["repair"], X_dr_by_split["repair"], thresholds_repair, alphas, policies, groups, args, args.model_random_seed + 311
        )
        outcome_cal = fit_outcome_bundle(
            data["repair"], X_dr_by_split["repair"], thresholds_cal, alphas, policies, groups, args, args.model_random_seed + 911
        )

        pred_repair = {"repair": predict_outcomes(outcome_repair, X_dr_by_split["repair"], args)}
        pred_caltest = {
            "cal": predict_outcomes(outcome_cal, X_dr_by_split["cal"], args),
            "test": predict_outcomes(outcome_cal, X_dr_by_split["test"], args),
        }
    else:
        outcome_repair = None
        outcome_cal = None
        pred_repair = {}
        pred_caltest = {}

    all_manifests: list[dict[str, Any]] = []
    for estimator in estimators:
        print(f"  computing estimator={estimator} for score={score} ...", flush=True)
        curve_parts: list[pd.DataFrame] = []
        cell_parts: list[pd.DataFrame] = []
        if estimator in {"raw_curve", "ipw_overlap_curve"}:
            for split_name, thresholds in [("repair", thresholds_repair), ("cal", thresholds_cal), ("test", thresholds_cal)]:
                split = data[split_name]
                if estimator == "raw_curve":
                    weights = np.ones_like(np.asarray(split["g"], dtype=float))
                else:
                    weights = w_by_split[split_name]
                gf = group_curve(split, split_name, thresholds, alphas, meta, weights, policies)
                cf = cell_curve(split, split_name, thresholds, alphas, meta, weights, policies)
                gf["estimator"] = estimator
                cf["estimator"] = estimator
                gf["threshold_source"] = "repair" if split_name == "repair" else "cal"
                cf["threshold_source"] = "repair" if split_name == "repair" else "cal"
                curve_parts.append(gf)
                cell_parts.append(cf)
        else:
            for split_name, thresholds, preds in [
                ("repair", thresholds_repair, pred_repair["repair"]),
                ("cal", thresholds_cal, pred_caltest["cal"]),
                ("test", thresholds_cal, pred_caltest["test"]),
            ]:
                split = data[split_name]
                r = target_tilt(e_dr_by_split[split_name], args.target_tilt, args.propensity_clip)
                gf, cf = dr_group_cell_curves(
                    split=split,
                    split_name=split_name,
                    thresholds=thresholds,
                    alphas=alphas,
                    policies=policies,
                    meta=meta,
                    e=e_dr_by_split[split_name],
                    r=r,
                    positive_group=pos_group,
                    outcome_pred=preds,
                    estimator=estimator,
                    args=args,
                )
                curve_parts.append(gf)
                cell_parts.append(cf)
        curve = pd.concat(curve_parts, ignore_index=True)
        cell = pd.concat(cell_parts, ignore_index=True)
        score_root = mkdir(out_root / estimator / score)
        man = save_and_plot_estimator_score(estimator, score, score_root, curve, cell, args)
        all_manifests.append(man)
    # Save score-level outcome model manifests without huge model pickles by default.
    if bool(args.save_nuisance_models):
        nuisance_dir = mkdir(out_root / "nuisance" / "outcome" / score)
        with (nuisance_dir / "outcome_repair_thresholds.pkl").open("wb") as f:
            pickle.dump(outcome_repair, f)
        with (nuisance_dir / "outcome_cal_thresholds.pkl").open("wb") as f:
            pickle.dump(outcome_cal, f)
    return all_manifests


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BioBias double-robust standardized curve pipeline.")
    p.add_argument("--seed-dir", type=Path, default=None, help="Path to outputs_biobias_train/<run>/seed_X")
    p.add_argument("--run-root", type=Path, default=None, help="Path to outputs_biobias_train/<run>; use with --seed")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prob-dir", type=Path, default=None, help="Path directly containing probability artifacts")
    p.add_argument("--output-root", type=Path, default=None, help=f"Default: <seed_dir>/{DR_OUTPUT_DIRNAME}")
    p.add_argument("--learned-root", type=Path, default=None, help="Optional learned propensity root containing learned_propensity_model.pkl; if provided, skip propensity training.")
    p.add_argument("--clean-output", action="store_true")

    # Propensity features/model: kept compatible with learned_propensity.py.
    p.add_argument("--feature-mode", choices=["safe_main", "safe_main_len", "safe_topk", "safe_probs", "safe_probs_len", "safe_score_profile", "safe_score_profile_len"], default="safe_main")
    p.add_argument("--model-type", choices=["logistic", "histgb", "mlp", "torch_mlp", "auto"], default="logistic",
                   help="Propensity model for e(H)=P(G=positive_group|H).")
    p.add_argument("--auto-candidates", default="logistic,torch_mlp")
    p.add_argument("--positive-group-name", default="Male")
    p.add_argument("--weight-scheme", choices=["ipw", "overlap", "matching"], default="overlap")
    p.add_argument("--propensity-clip", type=float, default=0.02)
    p.add_argument("--trim-quantile", type=float, default=0.995)
    p.add_argument("--max-weight", type=float, default=None)
    p.add_argument("--top-k-prob-features", type=int, default=3)
    p.add_argument("--score-profile-grid", default=DEFAULT_SCORE_PROFILE_GRID)
    p.add_argument("--simple-score-grid", default=DEFAULT_SIMPLE_SCORE_GRID)
    p.add_argument("--saps-raps-score-grid", default=DEFAULT_SAPS_RAPS_SCORE_GRID)
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--histgb-max-iter", type=int, default=250)
    p.add_argument("--histgb-learning-rate", type=float, default=0.04)
    p.add_argument("--histgb-l2", type=float, default=0.03)
    p.add_argument("--histgb-max-leaf-nodes", type=int, default=15)
    p.add_argument("--mlp-hidden-layers", default="64,32")
    p.add_argument("--mlp-alpha", type=float, default=1e-3)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-max-iter", type=int, default=250)
    p.add_argument("--torch-mlp-epochs", type=int, default=40)
    p.add_argument("--torch-mlp-batch-size", type=int, default=4096)
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--mlp-validation-fraction", type=float, default=0.15)
    p.add_argument("--mlp-n-iter-no-change", type=int, default=12)
    p.add_argument("--disable-mlp-early-stopping", action="store_true")
    p.add_argument("--repair-val-frac", type=float, default=0.25)
    p.add_argument("--selection-max-smd-weight", type=float, default=0.25)
    p.add_argument("--selection-ess-weight", type=float, default=0.50)
    p.add_argument("--model-random-seed", type=int, default=0)

    # Strict common-L design for OR/AIPW.  IPW can still use the richer --feature-mode.
    p.add_argument("--dr-feature-mode", choices=["same_as_ipw", "balanced_l_dr", "lean_difficulty", "balanced_score_profile"], default="same_as_ipw",
                   help="Common adjustment features L_DR for OR/AIPW. Use balanced_l_dr to avoid outcome set-size proxies.")
    p.add_argument("--dr-model-type", choices=["logistic", "histgb", "mlp"], default="logistic",
                   help="Propensity model for e_L(L_DR) used inside AIPW. This is separate from the main rich-H IPW model.")
    p.add_argument("--dr-top-label-k", type=int, default=2,
                   help="Number of top predicted label identities included in balanced_l_dr.")
    p.add_argument("--dr-include-coarse-bins", action="store_true", default=True,
                   help="Include fixed coarse bins of difficulty summaries in balanced_l_dr.")
    p.add_argument("--no-dr-coarse-bins", dest="dr_include_coarse_bins", action="store_false")
    p.add_argument("--forbid-outcome-setsize-proxies", action="store_true",
                   help="Reject OR/AIPW feature choices that expose full probabilities or score-profile/set-size fingerprints to the outcome model.")
    p.add_argument("--dr-logistic-c", type=float, default=1.0)
    p.add_argument("--dr-logistic-max-iter", type=int, default=1000)
    p.add_argument("--dr-histgb-max-iter", type=int, default=160)
    p.add_argument("--dr-histgb-learning-rate", type=float, default=0.04)
    p.add_argument("--dr-histgb-l2", type=float, default=0.03)
    p.add_argument("--dr-histgb-max-leaf-nodes", type=int, default=15)
    p.add_argument("--dr-mlp-hidden-layers", default="32,16")
    p.add_argument("--dr-mlp-alpha", type=float, default=1e-3)
    p.add_argument("--dr-mlp-lr", type=float, default=1e-3)
    p.add_argument("--dr-mlp-max-iter", type=int, default=250)

    # Outcome regression for OR and AIPW.
    p.add_argument("--outcome-design", choices=["group_specific", "pooled_additive", "pooled_group_aware"], default="group_specific",
                   help="Outcome nuisance structure. group_specific fits separate group models; pooled_additive uses group intercepts plus one common L_DR difficulty model; pooled_group_aware fits one pooled Q(A,L) model.")
    p.add_argument("--outcome-additive-intercept-shrinkage", type=float, default=0.0,
                   help="For --outcome-design pooled_additive, shrink group intercepts toward the pooled mean. Default 0.0 uses ordinary group means.")
    p.add_argument("--outcome-model", choices=["constant", "ridge", "histgb", "rf", "mlp"], default="ridge")
    p.add_argument("--min-outcome-n", type=int, default=30)
    p.add_argument("--outcome-ridge-alpha", type=float, default=1.0)
    p.add_argument("--outcome-histgb-max-iter", type=int, default=160)
    p.add_argument("--outcome-histgb-learning-rate", type=float, default=0.05)
    p.add_argument("--outcome-histgb-l2", type=float, default=0.03)
    p.add_argument("--outcome-histgb-max-leaf-nodes", type=int, default=15)
    p.add_argument("--outcome-rf-estimators", type=int, default=200)
    p.add_argument("--outcome-rf-min-samples-leaf", type=int, default=10)
    p.add_argument("--outcome-rf-max-depth", type=int, default=0)
    p.add_argument("--outcome-n-jobs", type=int, default=1)
    p.add_argument("--outcome-mlp-hidden-layers", default="128,64")
    p.add_argument("--outcome-mlp-alpha", type=float, default=1e-3)
    p.add_argument("--outcome-mlp-lr", type=float, default=1e-3)
    p.add_argument("--outcome-mlp-max-iter", type=int, default=250)
    p.add_argument("--outcome-monotone-project", action="store_true",
                   help="Post-process outcome predictions so estimated set-size and coverage curves are non-increasing in alpha.")
    p.add_argument("--target-tilt", choices=["overlap", "uniform"], default="overlap")
    p.add_argument("--estimators", default="raw_curve,ipw_overlap_curve,or_curve,aipw_dr_curve")
    p.add_argument("--save-nuisance-models", action="store_true")

    # Score/curve settings.
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
    p.add_argument("--no-curve-figures", action="store_true")
    p.add_argument("--write-sample-tables", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.mlp_early_stopping = not bool(args.disable_mlp_early_stopping)
    args.alpha_grid_values = parse_grid(args.alpha_grid)
    if float(args.alpha0) not in args.alpha_grid_values:
        args.alpha_grid_values = sorted(args.alpha_grid_values + [float(args.alpha0)])
    args.figure_formats_list = [x.strip().lower() for x in args.figure_formats.split(",") if x.strip()]
    estimators = parse_csv_list(args.estimators, ESTIMATOR_DIRS)
    need_dr_estimators = any(e in {"or_curve", "aipw_dr_curve"} for e in estimators)
    validate_dr_feature_safety(args)

    seed_dir = resolve_seed_dir(args)
    prob_dir = resolve_prob_dir(args, seed_dir)
    if args.output_root is None:
        out_root = seed_dir / DR_OUTPUT_DIRNAME
    else:
        out_root = args.output_root.expanduser().resolve()
        if out_root.name.startswith("seed_"):
            out_root = out_root / DR_OUTPUT_DIRNAME
    if args.clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root = unique_output_path(out_root)
    mkdir(out_root)
    tables_root = mkdir(out_root / "tables")
    figs_root = mkdir(out_root / "figures")
    nuisance_root = mkdir(out_root / "nuisance")

    meta = load_meta(prob_dir, seed_dir)
    base_data = load_splits(prob_dir, ["repair", "cal", "test"])
    repair = base_data["repair"]
    pos_group = positive_group_id(meta, repair["g"], args.positive_group_name)

    print("Preparing deployment-safe propensity model")
    print(f"  e(H)=P(G={group_name(meta, pos_group)} | H), feature_mode={args.feature_mode}")
    print("  validity boundary: e(H), m(H) never enter CP scores/quantiles/prediction sets")
    if args.learned_root is not None:
        learned_root = args.learned_root.expanduser().resolve()
        model_path = learned_root / "learned_propensity_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing learned_propensity_model.pkl under --learned-root: {learned_root}")
        print(f"  loading existing learned propensity model from: {model_path}")
        prop_bundle = load_pickle(model_path)
    else:
        print("  training learned propensity model on D_repair")
        prop_bundle = fit_propensity_model(repair, args=args, positive_group=pos_group)
    prop_bundle["meta_summary"] = {
        "positive_group": int(pos_group),
        "positive_group_name": group_name(meta, pos_group),
        "feature_safety": "deployment-safe: features are constructed from frozen model probabilities only; true occupation is excluded",
    }
    save_pickle(prop_bundle, nuisance_root / "learned_propensity_model.pkl")
    jdump({k: v for k, v in prop_bundle.items() if k not in {"model", "scaler"}}, nuisance_root / "learned_propensity_model_manifest.json")
    pd.DataFrame(prop_bundle.get("selection_table", [])).to_csv(tables_root / "propensity_model_selection_on_repair_validation.csv", index=False)

    e_repair_for_trim_cap, _ = predict_propensity(prop_bundle, repair, args.propensity_clip)
    fixed_trim_cap = estimate_weight_trim_cap(
        repair["g"],
        e_repair_for_trim_cap,
        pos_group,
        args.weight_scheme,
        args.trim_quantile,
    )

    e_by_split: dict[str, np.ndarray] = {}
    w_by_split: dict[str, np.ndarray] = {}
    X_by_split: dict[str, np.ndarray] = {}
    traces: list[pd.DataFrame] = []
    need_trace = bool(args.write_sample_tables) or (not bool(args.no_curve_figures))
    balances: list[pd.DataFrame] = []
    cat_balances: list[pd.DataFrame] = []
    weight_summaries: list[pd.DataFrame] = []
    for split_name in ["repair", "cal", "test"]:
        print(f"  computing propensity features/weights for split={split_name} ...", flush=True)
        split = base_data[split_name]
        e, X_raw = predict_propensity(prop_bundle, split, args.propensity_clip)
        w = make_weights(
            split["g"],
            e,
            pos_group,
            args.weight_scheme,
            args.trim_quantile,
            args.max_weight,
            fixed_trim_cap=fixed_trim_cap,
        )
        e_by_split[split_name] = e
        w_by_split[split_name] = w
        X_by_split[split_name] = X_raw
        if need_trace:
            traces.append(make_weight_trace(split_name, split, e, w, meta, pos_group))
        balances.append(feature_balance_table(split_name, split["g"], X_raw, prop_bundle["feature_names"], w, meta))
        cat_balances.append(categorical_balance_table(split_name, split["g"], split["z"], w, meta, "true_occupation_posthoc"))
        weight_summaries.append(weight_summary_table(split_name, split["g"], e, w, meta))
        print(f"    done split={split_name}; n={len(split['g'])}, features={X_raw.shape[1]}", flush=True)

    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    balance = pd.concat(balances, ignore_index=True)
    cat_balance = pd.concat(cat_balances, ignore_index=True)
    weight_summary = pd.concat(weight_summaries, ignore_index=True)
    weight_summary.to_csv(tables_root / "propensity_weight_summary_repair_cal_test.csv", index=False)
    # Aliases consumed by the shared plotting functions.
    weight_summary.to_csv(tables_root / "learned_propensity_weight_summary_repair_cal_test.csv", index=False)
    balance.to_csv(tables_root / "balance_diagnostics_safe_features_repair_cal_test.csv", index=False)
    cat_balance.to_csv(tables_root / "balance_diagnostics_true_occupation_posthoc_repair_cal_test.csv", index=False)
    if not trace.empty:
        if args.write_sample_tables:
            trace.to_csv(tables_root / "learned_propensity_weights_repair_cal_test_sample_trace.csv", index=False)
        else:
            trace.groupby("split", group_keys=False).head(5000).to_csv(tables_root / "learned_propensity_weights_repair_cal_test_sample_trace_head5000_each.csv", index=False)
    if not bool(args.no_curve_figures):
        for split_name in ["repair", "cal", "test"]:
            plot_weight_distribution(trace, split_name, figs_root, args.figure_formats_list)
            plot_balance(balance, split_name, figs_root, args.figure_formats_list)

    # Build the strict common-L nuisances only when OR/AIPW estimators are requested.
    # In an IPW-only run, this avoids fitting or reporting unused outcome/DR
    # nuisance objects that could be mistaken for part of the IPW estimand.
    X_dr_by_split: dict[str, np.ndarray] = {}
    e_dr_by_split: dict[str, np.ndarray] = {}
    w_dr_by_split: dict[str, np.ndarray] = {}
    dr_feature_names: list[str] = []
    fixed_dr_trim_cap: float | None = None
    if need_dr_estimators:
        print("Preparing common-L_DR nuisance features for OR/AIPW")
        print(f"  dr_feature_mode={args.dr_feature_mode}; dr_model_type={args.dr_model_type}")
        for split_name in ["repair", "cal", "test"]:
            X_dr, names_dr = make_dr_features(base_data[split_name], args, X_by_split.get(split_name), prop_bundle.get("feature_names"))
            X_dr_by_split[split_name] = X_dr
            if not dr_feature_names:
                dr_feature_names = names_dr
            print(f"    L_DR split={split_name}: n={X_dr.shape[0]}, features={X_dr.shape[1]}", flush=True)

        if str(args.dr_feature_mode).lower() in {"same_as_ipw", "same_as_propensity", "same"}:
            print("  using main rich-H propensity as the AIPW propensity because dr_feature_mode=same_as_ipw")
            dr_prop_bundle = {
                "model_type": prop_bundle.get("model_type"),
                "feature_mode": args.dr_feature_mode,
                "feature_names": dr_feature_names,
                "positive_group": int(pos_group),
                "n_features": int(X_dr_by_split["repair"].shape[1]),
                "fixed_trim_cap_source_split": "repair",
                "fixed_trim_cap_raw_weight_scale": fixed_trim_cap,
                "note": "AIPW reuses the main IPW propensity; this is the full-profile/same-feature design.",
            }
            fixed_dr_trim_cap = fixed_trim_cap
            for split_name in ["repair", "cal", "test"]:
                e_dr_by_split[split_name] = e_by_split[split_name]
                w_dr_by_split[split_name] = w_by_split[split_name]
        else:
            print("  training e_L(L_DR) on D_repair for strict common-L AIPW")
            dr_prop_bundle = fit_feature_propensity_model(
                X_dr_by_split["repair"], repair["g"], pos_group, dr_feature_names, args, args.model_random_seed + 1701
            )
            e_dr_repair_for_trim_cap = predict_feature_propensity(dr_prop_bundle, X_dr_by_split["repair"], args.propensity_clip)
            fixed_dr_trim_cap = estimate_weight_trim_cap(
                repair["g"],
                e_dr_repair_for_trim_cap,
                pos_group,
                args.weight_scheme,
                args.trim_quantile,
            )
            for split_name in ["repair", "cal", "test"]:
                e_dr = predict_feature_propensity(dr_prop_bundle, X_dr_by_split[split_name], args.propensity_clip)
                w_dr = make_weights(
                    base_data[split_name]["g"],
                    e_dr,
                    pos_group,
                    args.weight_scheme,
                    args.trim_quantile,
                    args.max_weight,
                    fixed_trim_cap=fixed_dr_trim_cap,
                )
                e_dr_by_split[split_name] = e_dr
                w_dr_by_split[split_name] = w_dr

        save_pickle(dr_prop_bundle, nuisance_root / "dr_common_l_propensity_model.pkl")
        jdump({k: v for k, v in dr_prop_bundle.items() if k != "model"}, nuisance_root / "dr_common_l_propensity_model_manifest.json")

        dr_balances: list[pd.DataFrame] = []
        dr_cat_balances: list[pd.DataFrame] = []
        dr_weight_summaries: list[pd.DataFrame] = []
        for split_name in ["repair", "cal", "test"]:
            dr_balances.append(feature_balance_table(split_name, base_data[split_name]["g"], X_dr_by_split[split_name], dr_feature_names, w_dr_by_split[split_name], meta))
            dr_cat_balances.append(categorical_balance_table(split_name, base_data[split_name]["g"], base_data[split_name]["z"], w_dr_by_split[split_name], meta, "true_occupation_posthoc_dr_common_l"))
            dr_weight_summaries.append(weight_summary_table(split_name, base_data[split_name]["g"], e_dr_by_split[split_name], w_dr_by_split[split_name], meta))
        dr_balance = pd.concat(dr_balances, ignore_index=True)
        dr_cat_balance = pd.concat(dr_cat_balances, ignore_index=True)
        dr_weight_summary = pd.concat(dr_weight_summaries, ignore_index=True)
        dr_balance.to_csv(tables_root / "dr_balance_diagnostics_l_dr_features_repair_cal_test.csv", index=False)
        dr_cat_balance.to_csv(tables_root / "dr_balance_diagnostics_true_occupation_posthoc_repair_cal_test.csv", index=False)
        dr_weight_summary.to_csv(tables_root / "dr_propensity_weight_summary_repair_cal_test.csv", index=False)
    else:
        dr_prop_bundle = {
            "model_type": "not_used",
            "feature_mode": args.dr_feature_mode,
            "feature_names": [],
            "positive_group": int(pos_group),
            "n_features": 0,
            "note": "OR/AIPW estimators were not requested; common-L_DR nuisance models were not constructed.",
        }

    scores = ["simple", "aps", "saps", "raps"] if args.score == "all" else [args.score]
    manifests: list[dict[str, Any]] = []
    for sc in scores:
        print(f"Running double-robust curve pipeline for score={sc} ...", flush=True)
        manifests.extend(run_score_pipeline(
            sc, prob_dir, meta, X_by_split, e_by_split, w_by_split, X_dr_by_split, e_dr_by_split,
            pos_group, out_root, estimators, args
        ))

    comp, red = summarize_outputs(out_root, estimators, scores, args.diagnostic_policy)
    comp.to_csv(tables_root / "estimator_comparison_summary.csv", index=False)
    red.to_csv(tables_root / "reversal_reduction_summary.csv", index=False)

    manifest = {
        "script": "biobias_adjusted_curve_double_robust.py",
        "probability_dir": str(prob_dir),
        "seed_dir": str(seed_dir),
        "output_root": str(out_root),
        "nuisance_training_split": "repair",
        "repair_threshold_source_split": "repair",
        "cal_test_threshold_source_split": "cal",
        "reporting_split": "test",
        "positive_group": int(pos_group),
        "positive_group_name": group_name(meta, pos_group),
        "ipw_feature_mode": args.feature_mode,
        "dr_feature_mode": args.dr_feature_mode,
        "dr_model_type": args.dr_model_type,
        "dr_n_features": int(X_dr_by_split["repair"].shape[1]) if need_dr_estimators else 0,
        "fixed_trim_cap_source_split": "repair" if fixed_trim_cap is not None else None,
        "fixed_trim_cap_raw_weight_scale": fixed_trim_cap,
        "fixed_trim_cap_note": "Raw-weight trimming cap estimated once on D_repair and reused for repair/cal/test.",
        "fixed_dr_trim_cap_source_split": "repair" if fixed_dr_trim_cap is not None else None,
        "fixed_dr_trim_cap_raw_weight_scale": fixed_dr_trim_cap,
        "outcome_nuisance_used": bool(need_dr_estimators),
        "dr_feature_safety": "L_DR uses frozen probability-difficulty summaries only and excludes full probability vectors, score-count fingerprints, calibration thresholds, alpha-specific set-size proxies, true labels, and true occupations when balanced_l_dr is selected.",
        "forbid_outcome_setsize_proxies": bool(args.forbid_outcome_setsize_proxies),
        "feature_safety": "deployment-safe: H/L_DR use frozen model probability features only; true occupation is used only for post-hoc audit grouping",
        "validity_boundary": "The conformal score, unweighted D_cal calibration quantile, and prediction-set membership rule are unchanged. Propensity and outcome models are used only after prediction sets are formed, to estimate standardized curve summaries.",
        "not_a_weighted_conformal_method": True,
        "propensity_model_type_requested": args.model_type,
        "propensity_model_type_selected": prop_bundle.get("model_type"),
        "dr_propensity_model_type_selected": dr_prop_bundle.get("model_type"),
        "ipw_design": "rich-H IPW curve uses --feature-mode and learned overlap weights",
        "or_aipw_design": "OR and AIPW use the common L_DR defined by --dr-feature-mode; AIPW uses e_L(L_DR) and m_g,alpha(L_DR)",
        "outcome_feature_restrictions": [
            "no full probability vector in balanced_l_dr",
            "no full sorted probability vector in balanced_l_dr",
            "no dense cumulative-mass curve in balanced_l_dr",
            "no score-count/set-size fingerprints in balanced_l_dr",
            "no calibration quantile or threshold features",
            "no alpha-specific predicted set-size proxy",
            "no true occupation",
            "no true label"
        ],
        "weight_scheme": args.weight_scheme,
        "target_tilt": args.target_tilt,
        "outcome_model": args.outcome_model,
        "outcome_design": args.outcome_design,
        "outcome_design_note": "group_specific fits separate group models; pooled_additive uses group intercepts plus one common L_DR difficulty model and forbids group-by-L outcome interactions; pooled_group_aware fits one pooled Q(A,L) model on L_DR plus group indicators",
        "estimators": estimators,
        "scores": scores,
        "alpha_grid": args.alpha_grid_values,
        "diagnostic_policy": args.diagnostic_policy,
        "score_manifests": manifests,
    }
    jdump(manifest, out_root / "double_robust_manifest.json")
    # Also write a familiar manifest name.
    jdump(manifest, out_root / "analysis_manifest.json")
    print(f"Done. Output root: {out_root}")
    if not comp.empty:
        print("\nEstimator comparison summary:")
        print(comp.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
