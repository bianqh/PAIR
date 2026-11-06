from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaselineConfig:
    dataset_json: str = "datasets/RSITMD/dataset_RSITMD.json"
    image_root: str = ""
    output_dir: str = "outputs/rsitmd_clip_qwen_progressive"
    train_split: str = "train"
    eval_split: str = "test"

    model_type: str = "clip_qwen"
    pretrained_weight_path: str = "weights/RS5M_ViT-B-32_RET-2.pt"
    qwen_model_name_or_path: str = "weights"
    qwen_weight_path: str = "weights/best_student.pt"
    checkpoint: str = ""

    model_name: str = "ViT-B-32"
    text_source: str = "sentence"
    loss_name: str = "itc"

    image_size: int = 224
    max_text_length: int = 256
    projection_dim: int = 512
    qwen_text_pooling: str = "last"

    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 8
    epochs: int = 20
    lr: float = 1e-5
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    warmup_epochs: float = 2.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    amp: bool = True
    precision: str = "bf16"
    allow_tf32: bool = True

    seed: int = 42
    grad_clip_norm: float = 1.0
    log_interval: int = 20
    eval_interval: int = 1
    device: str = "cuda"
    pin_memory: bool = True

    chunk_size: int = 1024
    topk: int = 10
    save_topk: bool = True
    save_embeddings: bool = False

    logit_scale_max: float = 50.0

    use_progressive_dialogue: bool = True
    max_progressive_rounds: int = 4
    eval_progressive_rounds: int = 4
    progressive_dialogue_seed: int = 42

    # New dataset-native negation settings.
    # If use_dataset_neg_attrs=True, hard_negative_content / neg_attrs in dataset_json
    # will be loaded directly as sample["neg_attrs"].
    use_dataset_neg_attrs: bool = True

    # During training, with probability negation_train_prob, append a negation sentence.
    # Once triggered, randomly sample 1~3 negation phrases by default.
    negation_train_prob: float = 0.5
    negation_train_min_attrs: int = 1
    negation_train_max_attrs: int = 3

    # During evaluation, use the first N negation phrases in dataset order.
    # 1 means using the first one; 2 means using the first and second.
    negation_eval_topn: int = 1

    debug: bool = False
    debug_train_samples: int = 256
    debug_eval_images: int = 64
    debug_eval_texts: int = 320
    debug_epochs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_tuple_value(field_name: str, value: Any) -> Any:
    if field_name == "betas" and isinstance(value, list):
        return tuple(float(v) for v in value)
    return value


def _merge_config_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    valid_fields = {field.name for field in fields(BaselineConfig)}
    merged = dict(base)
    for key, value in updates.items():
        if key not in valid_fields:
            raise KeyError(f"Unknown config field: {key}")
        merged[key] = _coerce_tuple_value(key, value)
    return merged


def load_config(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> BaselineConfig:
    config_dict = BaselineConfig().to_dict()
    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Config file must contain a mapping, got {type(loaded).__name__}.")
        config_dict = _merge_config_dict(config_dict, loaded)
    if overrides:
        config_dict = _merge_config_dict(config_dict, overrides)
    return BaselineConfig(**config_dict)
