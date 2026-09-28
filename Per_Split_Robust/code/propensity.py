"""Dataset-specific propensity and overlap-weight adapters."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DatasetUnit, SplitData
from .metrics import effective_sample_size
from .source_registry import PACKAGE_ROOT


@dataclass
class PropensityResult:
    repair_weights: np.ndarray
    test_weights: np.ndarray
    diagnostics: dict[str, Any]


def safe_auc(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else float("nan")


def binary_diagnostics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return {
        "auc": safe_auc(y, clipped),
        "log_loss": float(log_loss(y, np.column_stack((1.0 - clipped, clipped)), labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
        "propensity_min": float(np.min(clipped)),
        "propensity_p05": float(np.quantile(clipped, 0.05)),
        "propensity_p50": float(np.quantile(clipped, 0.50)),
        "propensity_p95": float(np.quantile(clipped, 0.95)),
        "propensity_max": float(np.max(clipped)),
    }


def load_biobias_utils() -> Any:
    source_dir = str(PACKAGE_ROOT / "biobias")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    return importlib.import_module("biobias_curve_utils")


def biobias_args() -> argparse.Namespace:
    utils = load_biobias_utils()
    return argparse.Namespace(
        feature_mode="safe_topk",
        top_k_prob_features=3,
        score_profile_grid=utils.DEFAULT_SCORE_PROFILE_GRID,
        simple_score_grid=utils.DEFAULT_SIMPLE_SCORE_GRID,
        saps_raps_score_grid=utils.DEFAULT_SAPS_RAPS_SCORE_GRID,
        model_type="torch_mlp",
        auto_candidates="logistic,torch_mlp",
        propensity_clip=0.03,
        weight_scheme="overlap",
        trim_quantile=0.95,
        max_weight=3.0,
        selection_max_smd_weight=0.25,
        selection_ess_weight=0.50,
        model_random_seed=0,
        repair_val_frac=0.25,
        logistic_c=1.0,
        histgb_max_iter=250,
        histgb_learning_rate=0.04,
        histgb_l2=0.03,
        histgb_max_leaf_nodes=15,
        mlp_hidden_layers="128,64",
        mlp_alpha=3e-3,
        mlp_lr=1e-3,
        mlp_max_iter=250,
        mlp_early_stopping=True,
        mlp_validation_fraction=0.15,
        mlp_n_iter_no_change=12,
        torch_mlp_epochs=60,
        torch_mlp_batch_size=4096,
        torch_device="auto",
        torch_num_threads=1,
    )


def fit_biobias_propensity(repair: SplitData, test: SplitData) -> PropensityResult:
    utils = load_biobias_utils()
    args = biobias_args()
    repair_dict = repair.as_original_dict()
    test_dict = test.as_original_dict()
    bundle = utils.fit_propensity_model(repair_dict, args=args, positive_group=0)
    repair_e, _ = utils.predict_propensity(bundle, repair_dict, args.propensity_clip)
    test_e, _ = utils.predict_propensity(bundle, test_dict, args.propensity_clip)
    fixed_trim_cap = utils.estimate_weight_trim_cap(
        repair.g, repair_e, 0, args.weight_scheme, args.trim_quantile
    )
    repair_weights = utils.make_weights(
        repair.g,
        repair_e,
        0,
        args.weight_scheme,
        args.trim_quantile,
        args.max_weight,
        fixed_trim_cap=fixed_trim_cap,
    )
    test_weights = utils.make_weights(
        test.g,
        test_e,
        0,
        args.weight_scheme,
        args.trim_quantile,
        args.max_weight,
        fixed_trim_cap=fixed_trim_cap,
    )
    y_test = (test.g.astype(int) == 0).astype(int)
    return PropensityResult(
        repair_weights=repair_weights,
        test_weights=test_weights,
        diagnostics={
            "propensity_model_selected": bundle.get("model_type", "torch_mlp"),
            "propensity_model_source": "refit_on_repair",
            "propensity_feature_mode": "safe_topk",
            "positive_group": 0,
            "positive_group_name": "Male",
            "propensity_seed": 0,
            "propensity_clip": args.propensity_clip,
            "trim_quantile": args.trim_quantile,
            "fixed_trim_cap": float(fixed_trim_cap),
            "max_weight": args.max_weight,
            "repair_ess_fraction": effective_sample_size(repair_weights) / len(repair_weights),
            "test_ess_fraction": effective_sample_size(test_weights) / len(test_weights),
            "test_diagnostics": binary_diagnostics(y_test, test_e),
        },
    )


def probability_profile_features(probabilities: np.ndarray, top_k: int = 3) -> tuple[np.ndarray, list[str]]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    n, n_classes = p.shape
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    k = max(1, min(int(top_k), n_classes))
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for rank in range(k):
        one_hot = np.zeros((n, n_classes), dtype=float)
        one_hot[np.arange(n), order[:, rank]] = 1.0
        blocks.append(one_hot)
        names.extend(f"top{rank + 1}_label_{label}" for label in range(n_classes))
    blocks.append(sorted_p[:, :k])
    names.extend(f"p_top{rank + 1}" for rank in range(k))
    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1] if n_classes >= 2 else np.zeros(n)
    p3 = sorted_p[:, 2] if n_classes >= 3 else np.zeros(n)
    entropy = -np.sum(p * np.log(p), axis=1) / max(math.log(n_classes), 1e-12)
    continuous = np.column_stack((entropy, p1 - p2, p2 - p3, sorted_p[:, : min(3, n_classes)].sum(axis=1)))
    blocks.append(continuous)
    names.extend(("entropy_norm", "margin12", "margin23", "top3_mass"))
    features = np.column_stack(blocks).astype(float)
    features[~np.isfinite(features)] = 0.0
    return features, names


def apply_folktables_caps(weights: np.ndarray, trim_cap: float, max_weight: float) -> np.ndarray:
    capped = np.minimum(np.asarray(weights, dtype=float), float(trim_cap))
    capped = np.minimum(capped, float(max_weight))
    finite = capped[np.isfinite(capped) & (capped >= 0)]
    median = float(np.median(finite)) if len(finite) else 1.0
    if median > 0:
        capped = capped / median
    capped = np.minimum(capped, float(max_weight))
    capped[~np.isfinite(capped)] = 0.0
    return capped


def fit_folktables_propensity(repair: SplitData, test: SplitData) -> PropensityResult:
    seed = 20260609
    repair_features, feature_names = probability_profile_features(repair.probs, top_k=3)
    test_features, _ = probability_profile_features(test.probs, top_k=3)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )
    model.fit(repair_features, repair.g.astype(int))
    repair_e = model.predict_proba(repair_features)[:, 1]
    test_e = model.predict_proba(test_features)[:, 1]
    clip = 0.03
    repair_e_clipped = np.clip(repair_e, clip, 1.0 - clip)
    test_e_clipped = np.clip(test_e, clip, 1.0 - clip)
    raw_repair = np.where(repair.g.astype(int) == 1, 1.0 - repair_e_clipped, repair_e_clipped)
    raw_test = np.where(test.g.astype(int) == 1, 1.0 - test_e_clipped, test_e_clipped)
    trim_cap = float(np.quantile(raw_repair[np.isfinite(raw_repair)], 0.95))
    repair_weights = apply_folktables_caps(raw_repair, trim_cap, 3.0)
    test_weights = apply_folktables_caps(raw_test, trim_cap, 3.0)
    return PropensityResult(
        repair_weights=repair_weights,
        test_weights=test_weights,
        diagnostics={
            "propensity_model_selected": "StandardScaler+LogisticRegression",
            "propensity_model_source": "refit_on_resplit_repair",
            "propensity_feature_mode": "safe_topk_probability_profile",
            "feature_count": len(feature_names),
            "positive_group": 1,
            "propensity_seed": seed,
            "propensity_clip": clip,
            "trim_quantile": 0.95,
            "fixed_trim_cap": trim_cap,
            "max_weight": 3.0,
            "repair_ess_fraction": effective_sample_size(repair_weights) / len(repair_weights),
            "test_ess_fraction": effective_sample_size(test_weights) / len(test_weights),
            "test_diagnostics": binary_diagnostics(test.g.astype(int), test_e),
        },
    )


def atus_probability_profile(probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(probabilities, dtype=float)
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    p1 = sorted_p[:, 0]
    p2 = sorted_p[:, 1]
    entropy = -(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return np.column_stack((p1, p2, p1 - p2, entropy)), ["p_top1", "p_top2", "margin", "entropy"]


def atus_weights(groups: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    clipped = np.clip(probabilities, 0.03, 0.97)
    target = groups.astype(str) == "female"
    raw = np.where(target, 1.0 - clipped, clipped)
    trim_cap = float(np.quantile(raw, 0.95))
    return np.minimum(raw, min(trim_cap, 3.0)), trim_cap


def fit_atus_propensity(repair: SplitData, test: SplitData) -> PropensityResult:
    seed = 4
    repair_features, feature_names = atus_probability_profile(repair.probs)
    test_features, _ = atus_probability_profile(test.probs)
    target_repair = (repair.g.astype(str) == "female").astype(int)
    target_test = (test.g.astype(str) == "female").astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(repair_features, target_repair)
    repair_e = model.predict_proba(repair_features)[:, 1]
    test_e = model.predict_proba(test_features)[:, 1]
    repair_weights, repair_cap = atus_weights(repair.g, repair_e)
    test_weights, test_cap = atus_weights(test.g, test_e)
    return PropensityResult(
        repair_weights=repair_weights,
        test_weights=test_weights,
        diagnostics={
            "propensity_model_selected": "StandardScaler+LogisticRegression",
            "propensity_model_source": "refit_on_resplit_repair",
            "propensity_feature_mode": "p_top1,p_top2,margin,entropy",
            "feature_count": len(feature_names),
            "positive_group": "female",
            "propensity_seed": seed,
            "propensity_clip": 0.03,
            "trim_quantile": 0.95,
            "repair_split_trim_cap": repair_cap,
            "test_split_trim_cap": test_cap,
            "trim_policy_note": "Canonical ATUS computes the 0.95 cap separately on each evaluation split.",
            "max_weight": 3.0,
            "repair_ess_fraction": effective_sample_size(repair_weights) / len(repair_weights),
            "test_ess_fraction": effective_sample_size(test_weights) / len(test_weights),
            "test_diagnostics": binary_diagnostics(target_test, test_e),
        },
    )


def fit_propensity(
    unit: DatasetUnit,
    repair: SplitData,
    test: SplitData,
) -> PropensityResult:
    if unit.dataset == "biobias":
        return fit_biobias_propensity(repair, test)
    if unit.dataset.startswith("folktables"):
        return fit_folktables_propensity(repair, test)
    if unit.dataset == "atus":
        return fit_atus_propensity(repair, test)
    raise ValueError(f"Unsupported dataset: {unit.dataset}")
