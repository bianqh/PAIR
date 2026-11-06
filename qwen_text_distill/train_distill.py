from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

csv.field_size_limit(sys.maxsize)

from clip_rsitmd_baseline.model import OpenCLIPRetrievalModel, _load_qwen_base_model, _load_transformers_classes
from clip_rsitmd_baseline.utils import (
    AverageMeter,
    build_timestamped_output_dir,
    configure_tf32,
    ensure_dir,
    get_autocast_context,
    get_timestamp_string,
    resolve_device,
    resolve_precision,
    save_config,
    save_json,
    set_seed,
    setup_logger,
)


@dataclass
class DistillConfig:
    output_dir: str = "outputs/qwen_text_distill"
    teacher_model_name: str = "ViT-B-32"
    teacher_weight_path: str = "weights/RS5M_ViT-B-32_RET-2.pt"
    qwen_model_name_or_path: str = "weights"
    qwen_pretrained_weight_path: str = "weights/qwen0.6B.safetensors"
    train_csv: list[str] = None
    val_csv: list[str] = None
    text_columns: list[str] = None
    max_text_length: int = 77
    qwen_text_pooling: str = "last"
    projection_dim: int = 512
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 8
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.05
    amp: bool = True
    precision: str = "bf16"
    allow_tf32: bool = True
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 200
    eval_interval: int = 1
    eval_steps: int = 2000
    max_eval_batches: int = 200
    grad_clip_norm: float = 1.0
    debug: bool = False
    debug_train_samples: int = 2048
    debug_val_samples: int = 1024
    debug_epochs: int = 1

    def __post_init__(self) -> None:
        if self.train_csv is None:
            self.train_csv = ["datasets/pub11_train_metadata.csv"]
        if self.val_csv is None:
            self.val_csv = ["datasets/pub11_validation_metadata.csv"]
        if self.text_columns is None:
            self.text_columns = ["text", "caption", "top1_cap_ft", "ssl_cap_ft", "top1_cap_vanilla", "ssl_cap_vanilla"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str, overrides: dict[str, Any] | None = None) -> DistillConfig:
    config_dict = DistillConfig().to_dict()
    with Path(config_path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    valid_fields = {field.name for field in fields(DistillConfig)}
    for key, value in loaded.items():
        if key not in valid_fields:
            raise KeyError(f"Unknown distill config field: {key}")
        if key == "betas" and isinstance(value, list):
            value = tuple(float(item) for item in value)
        config_dict[key] = value
    if overrides:
        config_dict.update(overrides)
    return DistillConfig(**config_dict)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def read_texts_from_csv(csv_paths: list[str], text_columns: list[str]) -> list[str]:
    texts: list[str] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            available_columns = [column for column in text_columns if column in reader.fieldnames]
            if not available_columns:
                raise ValueError(f"No configured text columns found in {path}. Columns: {reader.fieldnames}")
            for row in reader:
                for column in available_columns:
                    text = clean_text(row.get(column))
                    if text:
                        texts.append(text)
    return texts


class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def collate_texts(batch: list[str]) -> list[str]:
    return batch


def tokenized_to_device(tokenized_text, device: torch.device):
    if isinstance(tokenized_text, torch.Tensor):
        return tokenized_text.to(device, non_blocking=True)
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in tokenized_text.items()
    }


class QwenTextStudent(nn.Module):
    def __init__(
        self,
        qwen_model_name_or_path: str,
        qwen_pretrained_weight_path: str,
        max_text_length: int,
        projection_dim: int,
        qwen_text_pooling: str,
    ) -> None:
        super().__init__()
        if qwen_text_pooling not in {"mean", "last"}:
            raise ValueError(f"Unsupported qwen_text_pooling={qwen_text_pooling!r}. Expected 'mean' or 'last'.")
        self.max_text_length = max_text_length
        self.qwen_text_pooling = qwen_text_pooling

        _, _, AutoTokenizer = _load_transformers_classes()
        self.qwen = _load_qwen_base_model(qwen_model_name_or_path, qwen_pretrained_weight_path)
        text_width = int(self.qwen.config.hidden_size)

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_projection = nn.Linear(text_width, projection_dim, bias=False)
        nn.init.normal_(self.text_projection.weight, std=text_width**-0.5)

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

    def encode_text(self, tokenized_text, normalize: bool = True) -> torch.Tensor:
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


def build_scheduler(config: DistillConfig, optimizer: torch.optim.Optimizer, total_steps: int):
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def distill_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_features = F.normalize(student_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)
    cosine = (student_features * teacher_features).sum(dim=-1)
    cosine_loss = 1.0 - cosine
    mse = F.mse_loss(student_features, teacher_features, reduction="none").mean(dim=-1)
    return cosine_loss.mean(), mse.mean(), cosine.mean()


def save_student_checkpoint(path: Path, config: DistillConfig, student: QwenTextStudent, epoch: int, metrics: dict[str, float]) -> None:
    student_state = {
        key: value.detach().cpu()
        for key, value in student.state_dict().items()
        if key.startswith("qwen.")
    }
    student_state.update(
        {
            f"text_projection.{key}": value.detach().cpu()
            for key, value in student.text_projection.state_dict().items()
        }
    )
    torch.save(
        {
            "epoch": epoch,
            "model": student_state,
            "config": config.to_dict(),
            "metrics": metrics,
            "state_dict_type": "qwen_text_distill",
        },
        path,
    )


def run_eval(student, teacher, data_loader, device, precision_setup, max_batches: int = 0) -> dict[str, float]:
    student.eval()
    teacher.eval()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    cosine_meter = AverageMeter()
    autocast_context = get_autocast_context(precision_setup, device)
    with torch.no_grad():
        for batch_index, texts in enumerate(data_loader, start=1):
            if max_batches > 0 and batch_index > max_batches:
                break
            teacher_tokens = teacher.tokenize(texts).to(device, non_blocking=True)
            student_tokens = tokenized_to_device(student.tokenize(texts), device)
            with autocast_context:
                teacher_features = teacher.encode_text(teacher_tokens, normalize=True)
                student_features = student.encode_text(student_tokens, normalize=True)
                loss, mse, cosine = distill_loss(student_features, teacher_features)
            batch_size = len(texts)
            loss_meter.update(float(loss.detach().float().item()), n=batch_size)
            mse_meter.update(float(mse.detach().float().item()), n=batch_size)
            cosine_meter.update(float(cosine.detach().float().item()), n=batch_size)
    return {"loss": loss_meter.avg, "mse": mse_meter.avg, "cosine": cosine_meter.avg}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill RS5M CLIP text features into Qwen text encoder.")
    parser.add_argument("--config", required=True, help="Path to distillation YAML config.")
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Run a small smoke test.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.debug:
        overrides["debug"] = True
    config = load_config(args.config, overrides=overrides)
    if config.debug:
        config.epochs = config.debug_epochs
        config.batch_size = min(config.batch_size, 16)
        config.eval_batch_size = min(config.eval_batch_size, 32)
        config.num_workers = min(config.num_workers, 2)

    timestamp = get_timestamp_string()
    output_dir = ensure_dir(build_timestamped_output_dir(config.output_dir, timestamp=timestamp))
    logger = setup_logger(output_dir, f"distill_{timestamp}.log")
    save_config(config, output_dir)

    device = resolve_device(config.device)
    precision_setup = resolve_precision(config.amp, config.precision, device)
    configure_tf32(config.allow_tf32)
    set_seed(config.seed)

    logger.info("Distillation config saved to %s", output_dir / "config.yaml")
    logger.info("Using device=%s | requested_precision=%s | actual_precision=%s", device, config.precision, precision_setup.actual_precision)
    logger.info("%s", precision_setup.reason)

    train_texts = read_texts_from_csv(config.train_csv, config.text_columns)
    val_texts = read_texts_from_csv(config.val_csv, config.text_columns)
    if config.debug:
        train_texts = train_texts[: config.debug_train_samples]
        val_texts = val_texts[: config.debug_val_samples]
    logger.info("Loaded texts | train=%d | val=%d", len(train_texts), len(val_texts))

    train_loader = DataLoader(
        TextDataset(train_texts),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        collate_fn=collate_texts,
    )
    val_loader = DataLoader(
        TextDataset(val_texts),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        collate_fn=collate_texts,
    )

    teacher = OpenCLIPRetrievalModel(
        model_name=config.teacher_model_name,
        pretrained_weight_path=config.teacher_weight_path,
        max_text_length=77,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = QwenTextStudent(
        qwen_model_name_or_path=config.qwen_model_name_or_path,
        qwen_pretrained_weight_path=config.qwen_pretrained_weight_path,
        max_text_length=config.max_text_length,
        projection_dim=config.projection_dim,
        qwen_text_pooling=config.qwen_text_pooling,
    ).to(device)

    total_params, trainable_params_count = count_parameters(student)
    logger.info("Student parameters: total=%.2fM | trainable=%.2fM", total_params / 1e6, trainable_params_count / 1e6)

    trainable_params = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(config, optimizer, total_steps=max(len(train_loader) * config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=precision_setup.use_grad_scaler) if device.type == "cuda" else None

    best_cosine = float("-inf")
    history: list[dict[str, Any]] = []

    def evaluate_and_save(epoch: int, step: int | None = None, max_batches: int = 0) -> dict[str, float]:
        val_metrics = run_eval(student, teacher, val_loader, device, precision_setup, max_batches=max_batches)
        step_text = f" step {step}" if step is not None else ""
        logger.info(
            "Validation epoch %d%s | loss=%.5f | mse=%.6f | cosine=%.4f",
            epoch,
            step_text,
            val_metrics["loss"],
            val_metrics["mse"],
            val_metrics["cosine"],
        )
        nonlocal best_cosine
        if val_metrics["cosine"] > best_cosine:
            best_cosine = val_metrics["cosine"]
            save_student_checkpoint(output_dir / "best_student.pt", config, student, epoch, val_metrics)
            save_json(
                output_dir / "best_metrics.json",
                {"epoch": epoch, "step": step, "metrics": val_metrics, "max_eval_batches": max_batches},
            )
            logger.info("Updated best student checkpoint at epoch %d%s with cosine=%.4f", epoch, step_text, best_cosine)
        return val_metrics

    for epoch in range(1, config.epochs + 1):
        student.train()
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        cosine_meter = AverageMeter()
        autocast_context = get_autocast_context(precision_setup, device)

        for step, texts in enumerate(train_loader, start=1):
            teacher_tokens = teacher.tokenize(texts).to(device, non_blocking=True)
            student_tokens = tokenized_to_device(student.tokenize(texts), device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with autocast_context:
                    teacher_features = teacher.encode_text(teacher_tokens, normalize=True)

            with autocast_context:
                student_features = student.encode_text(student_tokens, normalize=True)
                loss, mse, cosine = distill_loss(student_features, teacher_features)

            if scaler is not None and precision_setup.use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
                optimizer.step()

            scheduler.step()

            batch_size = len(texts)
            loss_meter.update(float(loss.detach().float().item()), n=batch_size)
            mse_meter.update(float(mse.detach().float().item()), n=batch_size)
            cosine_meter.update(float(cosine.detach().float().item()), n=batch_size)

            if step % config.log_interval == 0 or step == len(train_loader):
                logger.info(
                    "Epoch %d Step %d/%d | lr=%.8f | loss=%.5f | mse=%.6f | cosine=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    optimizer.param_groups[0]["lr"],
                    loss_meter.avg,
                    mse_meter.avg,
                    cosine_meter.avg,
                )

            if config.eval_steps > 0 and step % config.eval_steps == 0:
                evaluate_and_save(epoch=epoch, step=step, max_batches=config.max_eval_batches)
                student.train()

        train_metrics = {"loss": loss_meter.avg, "mse": mse_meter.avg, "cosine": cosine_meter.avg}
        record: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

        if epoch % config.eval_interval == 0:
            val_metrics = evaluate_and_save(epoch=epoch, max_batches=0)
            record["val"] = val_metrics

        history.append(record)
        save_json(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
