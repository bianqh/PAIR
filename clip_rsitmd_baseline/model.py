from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must resolve to a state dict, got {type(checkpoint).__name__}.")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


class OpenCLIPRetrievalModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained_weight_path: str,
        image_size: int = 224,
        max_text_length: int = 77,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_text_length = max_text_length
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            self.model = open_clip.create_model(
                model_name=model_name,
                pretrained=None,
                load_weights=False,
                precision="fp32",
                force_image_size=image_size,
                force_context_length=max_text_length,
            )
        finally:
            logging.disable(previous_disable_level)
        self.tokenizer = open_clip.get_tokenizer(model_name, context_length=max_text_length)
        self.load_pretrained_weights(pretrained_weight_path)

    def load_pretrained_weights(self, weight_path: str) -> None:
        resolved = Path(weight_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Pretrained weight file not found: {resolved}")
        if resolved.suffix == ".safetensors":
            state_dict = load_safetensors_file(str(resolved))
        else:
            checkpoint = torch.load(str(resolved), map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Failed to load a valid state dict from {resolved}")
        cleaned_state_dict = {
            (key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Failed to cleanly load OpenCLIP weights from {resolved}.\n"
                f"Missing keys: {missing_keys[:20]}\nUnexpected keys: {unexpected_keys[:20]}"
            )

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts)

    def clamp_logit_scale(self, max_scale: float) -> None:
        max_log = torch.log(torch.tensor(max_scale, dtype=self.model.logit_scale.dtype, device=self.model.logit_scale.device))
        with torch.no_grad():
            self.model.logit_scale.clamp_(max=max_log.item())

    def get_logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.model.encode_image(images)
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def encode_text(self, tokenized_text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.model.encode_text(tokenized_text)
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def forward(self, images: torch.Tensor, tokenized_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(tokenized_text, normalize=True)
        logit_scale = self.get_logit_scale()
        return image_features, text_features, logit_scale


def _load_transformers_classes():
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "CLIP+Qwen retrieval requires the `transformers` package. "
            "Install a version that supports Qwen3, preferably transformers>=4.51.0."
        ) from exc
    return AutoConfig, AutoModel, AutoTokenizer


def _load_qwen_base_model(model_name_or_path: str, weight_path: str = ""):
    AutoConfig, AutoModel, _ = _load_transformers_classes()
    if not weight_path:
        model = AutoModel.from_pretrained(model_name_or_path)
        model.config.use_cache = False
        return model

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.use_cache = False
    model = AutoModel.from_config(config)
    resolved = Path(weight_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Qwen weight file not found: {resolved}")
    if resolved.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(resolved))
    else:
        checkpoint = torch.load(str(resolved), map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)

    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        if clean_key.startswith("lm_head."):
            continue
        cleaned_state_dict[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys while loading Qwen base weights from {resolved}: {unexpected_keys[:20]}"
        )
    important_missing = [
        key
        for key in missing_keys
        if not key.endswith("rotary_emb.inv_freq") and "rotary_emb" not in key
    ]
    if important_missing:
        raise RuntimeError(
            f"Missing keys while loading Qwen base weights from {resolved}: {important_missing[:20]}"
        )
    return model


def _init_qwen_base_model(model_name_or_path: str):
    AutoConfig, AutoModel, _ = _load_transformers_classes()
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.use_cache = False
    return AutoModel.from_config(config)


def _load_qwen_state_dict(weight_path: str) -> dict[str, torch.Tensor]:
    resolved = Path(weight_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Qwen weight file not found: {resolved}")
    if resolved.suffix == ".safetensors":
        return load_safetensors_file(str(resolved))
    checkpoint = torch.load(str(resolved), map_location="cpu")
    return _extract_state_dict(checkpoint)


def _detect_qwen_weight_kind(weight_path: str) -> str:
    state_dict = _load_qwen_state_dict(weight_path)
    tensor_keys = [key for key, value in state_dict.items() if isinstance(value, torch.Tensor)]
    if any(key.startswith("qwen.") or key.startswith("text_projection.") for key in tensor_keys):
        return "distilled_student"
    return "qwen_base"


def _build_identity_or_linear(in_features: int, out_features: int) -> nn.Linear:
    projection = nn.Linear(in_features, out_features, bias=False)
    if in_features == out_features:
        nn.init.eye_(projection.weight)
    else:
        nn.init.normal_(projection.weight, std=in_features**-0.5)
    return projection


class CLIPQwenRetrievalModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained_weight_path: str,
        qwen_model_name_or_path: str,
        qwen_weight_path: str = "",
        qwen_pretrained_weight_path: str = "",
        qwen_distill_checkpoint: str = "",
        image_size: int = 224,
        max_text_length: int = 128,
        projection_dim: int = 512,
        qwen_text_pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.qwen_model_name_or_path = qwen_model_name_or_path
        self.max_text_length = max_text_length
        self.projection_dim = projection_dim
        if qwen_text_pooling not in {"mean", "last"}:
            raise ValueError(f"Unsupported qwen_text_pooling={qwen_text_pooling!r}. Expected 'mean' or 'last'.")
        self.qwen_text_pooling = qwen_text_pooling

        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            clip_model = open_clip.create_model(
                model_name=model_name,
                pretrained=None,
                load_weights=False,
                precision="fp32",
                force_image_size=image_size,
            )
        finally:
            logging.disable(previous_disable_level)
        self.load_clip_pretrained_weights(clip_model, pretrained_weight_path)
        self.visual = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        qwen_base_weight_path = qwen_pretrained_weight_path
        qwen_student_weight_path = qwen_distill_checkpoint
        qwen_student_weight_kind = _detect_qwen_weight_kind(qwen_student_weight_path) if qwen_student_weight_path else ""

        if qwen_weight_path:
            qwen_weight_kind = _detect_qwen_weight_kind(qwen_weight_path)
            if qwen_weight_kind == "distilled_student":
                qwen_base_weight_path = ""
                qwen_student_weight_path = qwen_weight_path
                qwen_student_weight_kind = qwen_weight_kind
            else:
                qwen_base_weight_path = qwen_weight_path
                qwen_student_weight_path = ""
                qwen_student_weight_kind = ""

        _, _, AutoTokenizer = _load_transformers_classes()
        if qwen_student_weight_kind == "distilled_student" and not qwen_base_weight_path:
            self.qwen = _init_qwen_base_model(qwen_model_name_or_path)
        else:
            self.qwen = _load_qwen_base_model(qwen_model_name_or_path, qwen_base_weight_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        image_width = self._infer_clip_image_width()
        text_width = int(self.qwen.config.hidden_size)
        self.image_projection = _build_identity_or_linear(image_width, projection_dim)
        self.text_projection = nn.Linear(text_width, projection_dim, bias=False)
        nn.init.normal_(self.text_projection.weight, std=text_width**-0.5)
        if qwen_student_weight_path:
            self.load_qwen_distill_checkpoint(qwen_student_weight_path)

    def _infer_clip_image_width(self) -> int:
        if hasattr(self.visual, "output_dim"):
            return int(self.visual.output_dim)
        raise AttributeError("Could not infer OpenCLIP image feature width for image_projection.")

    @staticmethod
    def load_clip_pretrained_weights(clip_model: nn.Module, weight_path: str) -> None:
        resolved = Path(weight_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Pretrained CLIP weight file not found: {resolved}")
        if resolved.suffix == ".safetensors":
            state_dict = load_safetensors_file(str(resolved))
        else:
            checkpoint = torch.load(str(resolved), map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
        cleaned_state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
        missing_keys, unexpected_keys = clip_model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Failed to cleanly load OpenCLIP weights from {resolved}.\n"
                f"Missing keys: {missing_keys[:20]}\nUnexpected keys: {unexpected_keys[:20]}"
            )

    def load_qwen_distill_checkpoint(self, checkpoint_path: str) -> None:
        resolved = Path(checkpoint_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Qwen distill checkpoint not found: {resolved}")
        checkpoint = torch.load(str(resolved), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Qwen distill checkpoint must contain a state dict, got {type(state_dict).__name__}.")
        allowed_state = {
            key: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and (key.startswith("qwen.") or key.startswith("text_projection."))
        }
        if not allowed_state:
            raise RuntimeError(f"No qwen.* or text_projection.* tensors found in {resolved}.")
        missing_keys, unexpected_keys = self.load_state_dict(allowed_state, strict=False)
        important_missing = [
            key
            for key in missing_keys
            if key.startswith("qwen.") or key.startswith("text_projection.")
        ]
        if important_missing or unexpected_keys:
            raise RuntimeError(
                f"Failed to load Qwen distill checkpoint from {resolved}.\n"
                f"Missing keys: {important_missing[:20]}\nUnexpected keys: {unexpected_keys[:20]}"
            )

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

    def clamp_logit_scale(self, max_scale: float) -> None:
        max_log = torch.log(torch.tensor(max_scale, dtype=self.logit_scale.dtype, device=self.logit_scale.device))
        with torch.no_grad():
            self.logit_scale.clamp_(max=max_log.item())

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp()

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        image_features = self.visual(images)
        image_features = self.image_projection(image_features)
        if normalize:
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(self, tokenized_text, normalize: bool = True) -> torch.Tensor:
        if isinstance(tokenized_text, torch.Tensor):
            input_ids = tokenized_text
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]

        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden_states = outputs.last_hidden_state
        if self.qwen_text_pooling == "last":
            token_positions = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            pooled = hidden_states[batch_indices, token_positions]
        else:
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        text_features = self.text_projection(pooled)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images: torch.Tensor, tokenized_text) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(tokenized_text, normalize=True)
        logit_scale = self.get_logit_scale()
        return image_features, text_features, logit_scale


def build_retrieval_model(config) -> nn.Module:
    if config.model_type == "open_clip":
        return OpenCLIPRetrievalModel(
            model_name=config.model_name,
            pretrained_weight_path=config.pretrained_weight_path,
            image_size=config.image_size,
            max_text_length=config.max_text_length,
        )
    if config.model_type == "clip_qwen":
        return CLIPQwenRetrievalModel(
            model_name=config.model_name,
            pretrained_weight_path=config.pretrained_weight_path,
            qwen_model_name_or_path=config.qwen_model_name_or_path,
            qwen_weight_path=config.qwen_weight_path,
            qwen_pretrained_weight_path=config.qwen_pretrained_weight_path,
            qwen_distill_checkpoint=config.qwen_distill_checkpoint,
            image_size=config.image_size,
            max_text_length=config.max_text_length,
            projection_dim=config.projection_dim,
            qwen_text_pooling=config.qwen_text_pooling,
        )
    raise ValueError(f"Unsupported model_type={config.model_type!r}. Expected 'open_clip' or 'clip_qwen'.")
