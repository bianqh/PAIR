from __future__ import annotations

import json
import logging
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_dataset_name(dataset_json: str | Path) -> str:
    dataset_path = Path(dataset_json)
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset_name = str(payload.get("dataset", "")).strip()
        if dataset_name:
            return dataset_name
    except Exception:
        pass

    stem = dataset_path.stem
    for prefix in ("dataset_", "dataset"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix) :]
            break
    cleaned = stem.strip("_- ")
    return cleaned or "dataset"


def resolve_dataset_output_base(path: str | Path, dataset_name: str) -> Path:
    base_path = Path(path)
    normalized_dataset = str(dataset_name).strip() or "dataset"

    if any(part.lower() == normalized_dataset.lower() for part in base_path.parts):
        return base_path

    if base_path.parts and base_path.parts[0].lower() == "outputs":
        return Path(base_path.parts[0]) / normalized_dataset / Path(*base_path.parts[1:])

    return base_path / normalized_dataset


def build_timestamped_output_dir(path: str | Path, timestamp: str | None = None) -> Path:
    base_path = Path(path)
    run_timestamp = timestamp or get_timestamp_string()
    return base_path.parent / f"{run_timestamp}_{base_path.name}"


def find_latest_timestamped_output_dir(path: str | Path) -> Path | None:
    base_path = Path(path)
    pattern = f"*_{base_path.name}"
    candidates = [item for item in base_path.parent.glob(pattern) if item.is_dir()]
    if candidates:
        return max(candidates, key=lambda item: item.name)
    if base_path.exists():
        return base_path
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_tf32(allow_tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def setup_logger(output_dir: str | Path, filename: str) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger(f"clip_qwen.{Path(output_dir).resolve()}.{filename}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if logger.handlers:
        logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output_dir) / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_config(config: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir) / "config.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


@dataclass
class PrecisionSetup:
    requested_precision: str
    actual_precision: str
    use_amp: bool
    amp_dtype: torch.dtype | None
    use_grad_scaler: bool
    reason: str


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_precision(amp: bool, precision: str, device: torch.device) -> PrecisionSetup:
    if device.type != "cuda":
        return PrecisionSetup(
            requested_precision=precision,
            actual_precision="fp32",
            use_amp=False,
            amp_dtype=None,
            use_grad_scaler=False,
            reason="CUDA unavailable, using FP32 on CPU.",
        )

    precision = precision.lower()
    if not amp or precision == "fp32":
        return PrecisionSetup(
            requested_precision=precision,
            actual_precision="fp32",
            use_amp=False,
            amp_dtype=None,
            use_grad_scaler=False,
            reason="AMP disabled or FP32 requested.",
        )

    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return PrecisionSetup(
                requested_precision=precision,
                actual_precision="bf16",
                use_amp=True,
                amp_dtype=torch.bfloat16,
                use_grad_scaler=False,
                reason="CUDA BF16 supported, using BF16 autocast.",
            )
        return PrecisionSetup(
            requested_precision=precision,
            actual_precision="fp16",
            use_amp=True,
            amp_dtype=torch.float16,
            use_grad_scaler=True,
            reason="CUDA BF16 unsupported, falling back to FP16 autocast.",
        )

    if precision == "fp16":
        return PrecisionSetup(
            requested_precision=precision,
            actual_precision="fp16",
            use_amp=True,
            amp_dtype=torch.float16,
            use_grad_scaler=True,
            reason="Using FP16 autocast.",
        )

    raise ValueError(f"Unsupported precision={precision!r}. Expected one of 'bf16', 'fp16', 'fp32'.")


def get_autocast_context(precision_setup: PrecisionSetup, device: torch.device):
    if not precision_setup.use_amp or precision_setup.amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision_setup.amp_dtype)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.total = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.total += value * n
        self.count += n
        self.avg = self.total / max(self.count, 1)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    torch.save(state, Path(path))


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    strict: bool = True,
) -> dict[str, Any]:
    resolved = Path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Checkpoint load mismatch for {resolved}.\n"
            f"Missing keys: {missing_keys[:20]}\nUnexpected keys: {unexpected_keys[:20]}"
        )
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_metrics_for_log(metrics: dict[str, Any]) -> str:
    return (
        f"mean_recall={metrics['mean_recall']:.2f} | "
        f"t2i R@1/5/10={metrics['t2i']['R@1']:.2f}/{metrics['t2i']['R@5']:.2f}/{metrics['t2i']['R@10']:.2f} | "
        f"i2t R@1/5/10={metrics['i2t']['R@1']:.2f}/{metrics['i2t']['R@5']:.2f}/{metrics['i2t']['R@10']:.2f}"
    )
