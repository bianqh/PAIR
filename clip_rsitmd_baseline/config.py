from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaselineConfig:
    dataset_json: str = "datasets/RSITMD/dataset_RSITMD_with_content.json"
    image_root: str = ""
    output_dir: str = "outputs/rsitmd_vitb32_sdm"
    model_type: str = "open_clip"
    pretrained_weight_path: str = "weights/open_clip_ViT-B-32.safetensors"
    qwen_model_name_or_path: str = "Qwen/Qwen3-0.6B"
    qwen_weight_path: str = ""
    qwen_pretrained_weight_path: str = ""
    qwen_distill_checkpoint: str = ""
    checkpoint: str = ""
    model_name: str = "ViT-B-32"
    text_source: str = "sentence"
    loss_name: str = "sdm"
    image_size: int = 224
    max_text_length: int = 77
    projection_dim: int = 512
    qwen_text_pooling: str = "mean"
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 8
    epochs: int = 30
    lr: float = 4e-5
    min_lr_ratio: float = 0.1
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
    save_interval: int = 1
    device: str = "cuda"
    pin_memory: bool = True
    chunk_size: int = 1024
    topk: int = 10
    save_topk: bool = True
    save_embeddings: bool = False
    logit_scale_max: float = 100.0
    sdm_factor: float = 0.3
    sdm_epsilon: float = 1e-8
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
