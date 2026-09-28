#!/usr/bin/env python3
"""Train a BioBias classifier and export reusable probability arrays.

The script downloads Hugging Face datasets and models into a project-local
cache. It trains only on ``D_train`` and exports probabilities and metadata for
the train, calibration, repair, and test splits.

  D_train / D_cal / D_repair / D_test split
  -> train classifier on D_train
  -> save probabilities, labels, groups, audit cells, and sample identifiers
     for every split.

Conformal and curve analyses can then be rerun without retraining.
"""

from __future__ import annotations

# This experiment is PyTorch-only.  Some environments contain TensorFlow/Keras 3,
# which can make transformers.Trainer fail while importing optional TF modules.
# These variables must be set before importing transformers.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import random
import re
import time
from collections import Counter
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


def is_numeric_string(x: Any) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False


def canonical_profession_name(profession_id: int, raw_names: list[str] | None = None) -> str:
    if raw_names is not None and 0 <= int(profession_id) < len(raw_names):
        candidate = str(raw_names[int(profession_id)]).strip()
        if candidate and not is_numeric_string(candidate) and not candidate.startswith("label_"):
            return candidate
    if int(profession_id) in BIOBIAS_PROFESSION_ID_TO_NAME:
        return BIOBIAS_PROFESSION_ID_TO_NAME[int(profession_id)]
    return f"profession_{int(profession_id)}"



def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    path.write_text(json.dumps(conv(obj), indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def configure_hf_cache(args: argparse.Namespace) -> dict[str, str | bool]:
    """Place Hugging Face dataset/model caches under the project by default.

    Project-local caching keeps model and dataset downloads portable. The global
    Hugging Face cache remains available through the command-line option.
    """
    if args.use_global_hf_cache:
        return {"use_global_hf_cache": True}

    cache_dir = Path(args.hf_cache_dir).expanduser().resolve()
    hub_dir = cache_dir / "hub"
    datasets_dir = cache_dir / "datasets"
    assets_dir = cache_dir / "assets"
    for p in [cache_dir, hub_dir, datasets_dir, assets_dir]:
        mkdir(p)

    # These must be set before importing datasets/transformers/huggingface_hub.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_dir)
    os.environ["HF_ASSETS_CACHE"] = str(assets_dir)

    return {
        "use_global_hf_cache": False,
        "HF_HOME": str(cache_dir),
        "HF_HUB_CACHE": str(hub_dir),
        "HF_DATASETS_CACHE": str(datasets_dir),
        "HF_ASSETS_CACHE": str(assets_dir),
    }


def hf_hub_cache_dir(args: argparse.Namespace) -> str | None:
    if args.use_global_hf_cache:
        return None
    return str(Path(args.hf_cache_dir).expanduser().resolve() / "hub")


def hf_datasets_cache_dir(args: argparse.Namespace) -> str | None:
    if args.use_global_hf_cache:
        return None
    return str(Path(args.hf_cache_dir).expanduser().resolve() / "datasets")


def clean_text(x: str) -> str:
    x = str(x).replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", x).strip()


def nice_group_name(x: Any) -> str:
    z = str(x).strip()
    low = z.lower()
    if low in {"m", "male"}:
        return "Male"
    if low in {"f", "female"}:
        return "Female"
    return z.title() if z else z


def feature_names(feature: Any, values: list[int]) -> list[str]:
    names = getattr(feature, "names", None)
    if names is not None:
        return [str(x) for x in names]
    return [str(i) for i in range(int(max(values)) + 1)]


def profession_name_list(raw_names: list[str], profession_ids: list[int]) -> list[str]:
    return [canonical_profession_name(int(i), raw_names) for i in profession_ids]


def gender_name_list(raw_names: list[str]) -> list[str]:
    out = []
    for i, x in enumerate(raw_names):
        candidate = nice_group_name(x)
        if is_numeric_string(candidate) and i in BIOBIAS_GENDER_ID_TO_NAME:
            candidate = BIOBIAS_GENDER_ID_TO_NAME[i]
        out.append(candidate)
    if len(out) >= 2 and all(is_numeric_string(x) for x in out[:2]):
        out[0] = BIOBIAS_GENDER_ID_TO_NAME[0]
        out[1] = BIOBIAS_GENDER_ID_TO_NAME[1]
    return out


def load_biobias(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Load BioBias, keep top occupations, and attach stable sample IDs.

    The exported IDs make the downstream analysis stage auditable:
      sample_id      = "<original_split>:<row_index_within_original_split>"
      source_split   = original Hugging Face split name
      source_index   = row index within that original split
      artifact_index = row index after top-occupation filtering/concatenation

    These fields are never used for training; they are only saved for
    reproducibility and sample-level diagnosis tables.
    """
    from datasets import concatenate_datasets, load_dataset

    raw = load_dataset(args.dataset_name, cache_dir=hf_datasets_cache_dir(args))
    raw_profession_names = feature_names(raw["train"].features["profession"], raw["train"]["profession"])
    raw_gender_names = feature_names(raw["train"].features["gender"], raw["train"]["gender"])
    gender_names = gender_name_list(raw_gender_names)

    counts = Counter(map(int, raw["train"]["profession"]))
    old_labels = sorted([k for k, _ in counts.most_common(args.top_n_occupations)]) if args.top_n_occupations > 0 else sorted(counts)
    old_to_new = {old: new for new, old in enumerate(old_labels)}
    keep = set(old_labels)

    split_names = [s for s in ["train", "dev", "validation", "test"] if s in raw]
    tagged_splits = []
    for split_name in split_names:
        tagged = raw[split_name].map(
            lambda e, i, split_name=split_name: {
                "source_split": split_name,
                "source_index": int(i),
                "sample_id": f"{split_name}:{i}",
            },
            with_indices=True,
            load_from_cache_file=False,
        )
        tagged = tagged.filter(lambda e: int(e["profession"]) in keep, load_from_cache_file=False)
        tagged_splits.append(tagged)
    ds = concatenate_datasets(tagged_splits)

    if args.text_field in ds.column_names:
        text_col = args.text_field
    else:
        text_col = next(c for c in ["hard_text", "text", "bio", "description"] if c in ds.column_names)

    def map_row(e: dict[str, Any]) -> dict[str, Any]:
        y = old_to_new[int(e["profession"])]
        return {
            "text": clean_text(e[text_col]),
            "labels": y,
            "group": int(e["gender"]),
            "audit": y,  # BioBias audit cell is occupation; candidate-label convention uses z_y=y.
            "sample_id": str(e["sample_id"]),
            "source_split": str(e["source_split"]),
            "source_index": int(e["source_index"]),
        }

    ds = ds.map(map_row, remove_columns=ds.column_names, load_from_cache_file=False)
    ds = ds.map(lambda e, i: {"artifact_index": int(i)}, with_indices=True, load_from_cache_file=False)
    selected_profession_names = profession_name_list(raw_profession_names, old_labels)
    selected_professions = [
        {
            "compact_label": int(new_id),
            "profession_id": int(old_id),
            "occupation_name": str(selected_profession_names[new_id]),
        }
        for new_id, old_id in enumerate(old_labels)
    ]
    meta = {
        "dataset": args.dataset_name,
        "text_field": text_col,
        "old_profession_ids": old_labels,
        "label_names": selected_profession_names,
        "occupation_names": selected_profession_names,
        "profession_id_to_name": {str(i): canonical_profession_name(int(i), raw_profession_names) for i in sorted(set(old_labels) | set(BIOBIAS_PROFESSION_ID_TO_NAME))},
        "selected_professions": selected_professions,
        "gender_names": gender_names,
        "group_names": gender_names,
        "gender_id_to_name": {str(i): str(name) for i, name in enumerate(gender_names)},
        "num_labels": len(old_labels),
        "sample_id_definition": "<source_split>:<source_index>",
        "identity_fields": ["sample_id", "source_split", "source_index", "artifact_index"],
    }
    return ds, meta


def build_label_maps(meta: dict[str, Any]) -> dict[str, Any]:
    """Build portable occupation-name metadata for probability-only artifacts."""
    label_names = [str(x) for x in meta.get("label_names", [])]
    occupation_names = [str(x) for x in meta.get("occupation_names", label_names)]
    gender_names = [str(x) for x in meta.get("gender_names", [])]
    old_ids = [int(x) for x in meta.get("old_profession_ids", list(range(len(label_names))))]
    selected_professions = meta.get("selected_professions") or [
        {
            "compact_label": int(i),
            "profession_id": int(old_ids[i]),
            "occupation_name": str(occupation_names[i]),
        }
        for i in range(len(occupation_names))
    ]
    compact_to_old = {str(int(row["compact_label"])): int(row["profession_id"]) for row in selected_professions}
    compact_to_name = {str(int(row["compact_label"])): str(row["occupation_name"]) for row in selected_professions}
    old_to_new = {str(v): int(k) for k, v in compact_to_old.items()}
    profession_id_to_name = {str(k): str(v) for k, v in BIOBIAS_PROFESSION_ID_TO_NAME.items()}
    profession_id_to_name.update({str(row["profession_id"]): str(row["occupation_name"]) for row in selected_professions})
    return {
        "schema": "biobias_label_maps",
        "dataset": meta.get("dataset", "LabHC/bias_in_bios"),
        "audit_variable": "occupation",
        "audit_convention": "candidate_label",
        "z_equals_y": True,
        "num_labels": int(meta.get("num_labels", len(occupation_names))),
        "label_names": occupation_names,
        "occupation_names": occupation_names,
        "audit_names": occupation_names,
        "selected_professions": selected_professions,
        "profession_id_to_name": profession_id_to_name,
        "compact_label_to_profession_id": compact_to_old,
        "compact_label_to_occupation_name": compact_to_name,
        "new_label_to_old_profession_id": compact_to_old,
        "old_profession_id_to_new_label": old_to_new,
        "gender_names": gender_names,
        "group_names": gender_names,
        "gender_id_to_name": {str(i): str(name) for i, name in enumerate(gender_names)},
    }

def stratify_or_none(y: np.ndarray, g: np.ndarray) -> pd.Series | None:
    candidates = [
        pd.Series(g).astype(str) + "_" + pd.Series(y).astype(str),
        pd.Series(y).astype(str),
    ]
    for s in candidates:
        if not (s.value_counts() < 2).any():
            return s
    return None


def split_idx(idx: np.ndarray, y_all: np.ndarray, g_all: np.ndarray, train_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    strat = stratify_or_none(y_all[idx], g_all[idx])
    try:
        a, b = train_test_split(idx, train_size=train_size, random_state=seed, stratify=strat)
    except ValueError:
        a, b = train_test_split(idx, train_size=train_size, random_state=seed, stratify=None)
    return np.sort(a), np.sort(b)


def make_splits(ds: Any, seed: int, max_n: int | None) -> dict[str, Any]:
    y, g = np.asarray(ds["labels"]), np.asarray(ds["group"])
    idx = np.arange(len(ds))
    if max_n and max_n < len(ds):
        idx, _ = split_idx(idx, y, g, max_n / len(ds), seed)
        ds = ds.select(idx.tolist())
        y, g, idx = np.asarray(ds["labels"]), np.asarray(ds["group"]), np.arange(len(ds))

    train, rest = split_idx(idx, y, g, 0.40, seed)
    cal, rest = split_idx(rest, y, g, 0.25 / 0.60, seed + 1)
    repair, test = split_idx(rest, y, g, 0.25 / 0.35, seed + 2)
    return {
        "train": ds.select(train.tolist()),
        "cal": ds.select(cal.tolist()),
        "repair": ds.select(repair.tolist()),
        "test": ds.select(test.tolist()),
    }


def split_summary(splits: dict[str, Any], meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split_name, ds in splits.items():
        frame = pd.DataFrame({"group": ds["group"], "label": ds["labels"]})
        tab = frame.groupby(["group", "label"]).size().reset_index(name="n")
        tab.insert(0, "split", split_name)
        tab["group_name"] = tab["group"].map(lambda x: meta["gender_names"][int(x)])
        tab["label_name"] = tab["label"].map(lambda x: meta["label_names"][int(x)])
        rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def tokenize_splits(splits: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    def tok(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return {
        name: ds.map(tok, batched=True, remove_columns=[c for c in ds.column_names if c != "labels"], load_from_cache_file=False)
        for name, ds in splits.items()
    }


def configure_torch_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Configure CUDA runtime options for this PyTorch-only experiment.

    PyTorch may route scaled dot-product attention through fused CUDA/cuDNN
    kernels. Some CUDA/cuDNN combinations cannot initialize the fused SDPA
    planner, so the default uses the math backend unless automatic selection is
    requested explicitly.
    """
    import torch

    info: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        info["fp16_requested"] = bool(args.fp16)
        info["reproducible_defaults"] = not bool(args.fp16)
        info["tf32_enabled"] = not bool(args.disable_tf32)
        info["sdpa_backend"] = args.sdpa_backend
        try:
            torch.backends.cuda.matmul.allow_tf32 = not bool(args.disable_tf32)
            torch.backends.cudnn.allow_tf32 = not bool(args.disable_tf32)
            if hasattr(torch, "set_float32_matmul_precision") and not bool(args.disable_tf32):
                torch.set_float32_matmul_precision("high")

            if args.sdpa_backend == "math":
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(False)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                    torch.backends.cuda.enable_cudnn_sdp(False)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
                os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
        except Exception as exc:
            info["torch_runtime_warning"] = str(exc)
        print(f"CUDA device: {info['cuda_device_name_0']}")
        print(f"FP16 requested: {info['fp16_requested']}; TF32 enabled: {info['tf32_enabled']}")
        print(f"SDPA backend policy: {info['sdpa_backend']}")
    else:
        info["fp16_requested"] = False
        info["tf32_enabled"] = False
        info["sdpa_backend"] = "cpu"
        print("CUDA is not available. Training will run on CPU.")
    return info


def train_model(args: argparse.Namespace, splits: dict[str, Any], meta: dict[str, Any], seed_dir: Path) -> tuple[Any, dict[str, Any], float, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    torch_runtime = configure_torch_runtime(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=hf_hub_cache_dir(args))
    tokenized = tokenize_splits(splits, tokenizer, args.max_length)
    model_kwargs = dict(num_labels=meta["num_labels"], cache_dir=hf_hub_cache_dir(args))
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    except ValueError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)

    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    kw = dict(
        output_dir=str(mkdir(seed_dir / "model_tmp")),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        save_strategy="no",
        fp16=bool(args.fp16 and torch.cuda.is_available()),
    )
    kw["eval_strategy" if "eval_strategy" in allowed else "evaluation_strategy"] = "no"
    training_args = TrainingArguments(**{k: v for k, v in kw.items() if k in allowed})

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer_kw["processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kw)
    result = trainer.train()

    if args.save_model:
        model_dir = mkdir(seed_dir / "model")
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    return trainer, tokenized, float(result.training_loss), torch_runtime


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def export_probabilities(trainer: Any, tokenized: dict[str, Any], splits: dict[str, Any], out: Path, label_maps: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Export probabilities and all split-specific identifiers.

    Each split gets arrays such as probs_cal.npy, y_cal.npy, g_cal.npy, z_cal.npy,
    plus idx_cal.npy / sample_id_cal.npy / source_split_cal.npy / source_index_cal.npy.
    Stage 2 can therefore identify both the split role and the original sample.
    """
    mkdir(out)
    payload = {}
    for split_name in ["train", "cal", "repair", "test"]:
        pred = trainer.predict(tokenized[split_name])
        probs = softmax(np.asarray(pred.predictions))
        artifact_index = np.asarray(splits[split_name]["artifact_index"], dtype=int)
        y_arr = np.asarray(splits[split_name]["labels"], dtype=int)
        g_arr = np.asarray(splits[split_name]["group"], dtype=int)
        z_arr = np.asarray(splits[split_name]["audit"], dtype=int)
        compact_to_profession_id = {int(k): int(v) for k, v in label_maps.get("compact_label_to_profession_id", label_maps.get("new_label_to_old_profession_id", {})).items()}
        compact_to_occupation_name = {int(k): str(v) for k, v in label_maps.get("compact_label_to_occupation_name", {}).items()}
        group_names = [str(x) for x in label_maps.get("gender_names", [])]
        profession_id_arr = np.asarray([compact_to_profession_id.get(int(y), int(y)) for y in y_arr], dtype=int)
        occupation_name_arr = np.asarray([compact_to_occupation_name.get(int(y), f"occupation_{int(y)}") for y in y_arr], dtype=str)
        group_name_arr = np.asarray([group_names[int(g)] if 0 <= int(g) < len(group_names) else f"group_{int(g)}" for g in g_arr], dtype=str)
        z_profession_id_arr = np.asarray([compact_to_profession_id.get(int(z), int(z)) for z in z_arr], dtype=int)
        z_occupation_name_arr = np.asarray([compact_to_occupation_name.get(int(z), f"occupation_{int(z)}") for z in z_arr], dtype=str)
        payload[split_name] = {
            "probs": probs,
            "y": y_arr,
            "g": g_arr,
            "z": z_arr,
            "profession_id": profession_id_arr,
            "occupation_name": occupation_name_arr,
            "group_name": group_name_arr,
            "z_profession_id": z_profession_id_arr,
            "z_occupation_name": z_occupation_name_arr,
            "idx": artifact_index,
            "artifact_index": artifact_index,
            "sample_id": np.asarray(splits[split_name]["sample_id"], dtype=str),
            "source_split": np.asarray(splits[split_name]["source_split"], dtype=str),
            "source_index": np.asarray(splits[split_name]["source_index"], dtype=int),
        }
        for key, value in payload[split_name].items():
            np.save(out / f"{key}_{split_name}.npy", value)
    return payload


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BioBias classifier and export probabilities for curve/Simpson analysis.")
    p.add_argument("--dataset-name", default="LabHC/bias_in_bios")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--text-field", default="hard_text")
    p.add_argument("--output-root", type=Path, default=Path("outputs_biobias_train"))
    p.add_argument("--hf-cache-dir", type=Path, default=Path("cache/huggingface"), help="Local Hugging Face cache directory. Default: ./cache/huggingface")
    p.add_argument("--use-global-hf-cache", action="store_true", help="Do not set local HF_HOME/HF_* cache dirs; use the environment/global Hugging Face cache instead.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--top-n-occupations", type=int, default=10)
    p.add_argument("--max-total-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--fp16", action="store_true", help="Enable optional mixed-precision FP16 training; disabled by default.")
    p.add_argument("--disable-tf32", action="store_true", help="Disable TF32 matrix multiplication and convolution on supported NVIDIA GPUs.")
    p.add_argument("--sdpa-backend", choices=["math", "auto"], default="math", help="SDPA backend policy. The compatibility-oriented default is math.")
    p.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="eager", help="Transformers attention implementation hint. Default eager is conservative and avoids SDPA when supported.")
    p.add_argument("--save-model", action="store_true")
    return p


def main() -> None:
    args = args_parser().parse_args()
    cache_info = configure_hf_cache(args)
    seed_everything(args.seed)

    run = args.run_name or f"biobias_train_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = mkdir(args.output_root / run)
    seed_dir = mkdir(run_root / f"seed_{args.seed}")
    jdump({**vars(args), "hf_cache_runtime": cache_info}, run_root / "args.json")

    print("Loading BioBias and creating D_train/D_cal/D_repair/D_test = 40/25/25/10 ...")
    ds, meta = load_biobias(args)
    splits = make_splits(ds, args.seed, args.max_total_samples)
    split_sizes = {k: len(v) for k, v in splits.items()}
    print(split_sizes)
    print("Selected occupations:")
    for row in meta.get("selected_professions", []):
        print(f"  compact={row['compact_label']:2d}  profession_id={row['profession_id']:2d}  occupation={row['occupation_name']}")

    split_summary(splits, meta).to_csv(seed_dir / "split_summary_by_group_label.csv", index=False)
    label_maps = build_label_maps(meta)
    split_meta_payload = {
        "meta": meta,
        "label_maps": label_maps,
        "split_sizes": split_sizes,
        "split_ratio_target": {"D_train": 0.40, "D_cal": 0.25, "D_repair": 0.25, "D_test": 0.10},
        "hf_cache_runtime": cache_info,
        "saved_identifier_arrays": [
            "idx_{split}.npy",
            "artifact_index_{split}.npy",
            "sample_id_{split}.npy",
            "source_split_{split}.npy",
            "source_index_{split}.npy",
        ],
    }
    jdump(split_meta_payload, seed_dir / "split_meta.json")
    jdump(label_maps, seed_dir / "label_maps.json")
    pd.DataFrame(label_maps["selected_professions"]).to_csv(seed_dir / "label_maps.csv", index=False)

    print("Training classifier on D_train ...")
    trainer, tokenized, train_loss, torch_runtime = train_model(args, splits, meta, seed_dir)
    jdump({"train_loss": train_loss, "hf_cache_runtime": cache_info, "torch_runtime": torch_runtime}, seed_dir / "train_summary.json")

    print("Exporting probabilities for all splits ...")
    prob_dir = mkdir(seed_dir / "probabilities")
    jdump(label_maps, prob_dir / "label_maps.json")
    pd.DataFrame(label_maps["selected_professions"]).to_csv(prob_dir / "label_maps.csv", index=False)
    jdump(split_meta_payload, prob_dir / "split_meta.json")
    export_probabilities(trainer, tokenized, splits, prob_dir, label_maps)

    manifest = {
        "run_root": run_root,
        "seed_dir": seed_dir,
        "probabilities_dir": prob_dir,
        "split_meta": seed_dir / "split_meta.json",
        "split_summary": seed_dir / "split_summary_by_group_label.csv",
        "train_summary": seed_dir / "train_summary.json",
    }
    jdump(manifest, run_root / "manifest.json")
    jdump(manifest, seed_dir / "export_manifest.json")
    print(f"Done. Training artifact: {seed_dir}")


if __name__ == "__main__":
    main()
