from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .datasets import (
    RSITMDImageDataset,
    RSITMDTextDataset,
    RSITMDTrainDataset,
    build_dataset_sanity_report,
    load_rsitmd_splits,
)
from .engine import evaluate, train_one_epoch
from .losses import RetrievalLoss
from .model import build_retrieval_model
from .transforms import build_eval_transform, build_train_transform
from .utils import (
    build_timestamped_output_dir,
    configure_tf32,
    count_parameters,
    ensure_dir,
    format_metrics_for_log,
    get_timestamp_string,
    resolve_device,
    resolve_precision,
    save_checkpoint,
    save_config,
    save_json,
    set_seed,
    setup_logger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RSITMD CLIP baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--debug", action="store_true", help="Run a small debug training loop.")
    return parser


def build_optimizer(config, model):
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "logit_scale" in name:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )
    return optimizer


def build_scheduler(config, optimizer, steps_per_epoch: int):
    if config.scheduler != "cosine":
        raise ValueError(f"Unsupported scheduler={config.scheduler!r}. Only 'cosine' is implemented.")
    total_steps = max(int(config.epochs * steps_per_epoch), 1)
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        min_lr_scale = config.min_lr_ratio
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def apply_debug_settings(config):
    config.debug = True
    config.epochs = config.debug_epochs
    config.batch_size = min(config.batch_size, 16)
    config.eval_batch_size = min(config.eval_batch_size, 32)
    config.num_workers = min(config.num_workers, 2)
    if not str(config.output_dir).endswith("_debug"):
        config.output_dir = f"{config.output_dir}_debug"
    return config


def main() -> None:
    args = build_parser().parse_args()
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.debug:
        overrides["debug"] = True

    config = load_config(args.config, overrides=overrides)
    if config.debug:
        config = apply_debug_settings(config)

    run_timestamp = get_timestamp_string()
    output_dir = ensure_dir(build_timestamped_output_dir(config.output_dir, timestamp=run_timestamp))
    config.output_dir = str(output_dir)
    logger = setup_logger(output_dir, f"train_{run_timestamp}.log")
    save_config(config, output_dir)

    device = resolve_device(config.device)
    precision_setup = resolve_precision(config.amp, config.precision, device)
    configure_tf32(config.allow_tf32)
    set_seed(config.seed)

    logger.info("Training config saved to %s", output_dir / "config.yaml")
    logger.info("Using device=%s | requested_precision=%s | actual_precision=%s", device, config.precision, precision_setup.actual_precision)
    logger.info("%s", precision_setup.reason)

    sanity_report = build_dataset_sanity_report(config.dataset_json, image_root=config.image_root, text_source=config.text_source)
    logger.info("Dataset sanity: %s", sanity_report)

    splits = load_rsitmd_splits(
        dataset_json=config.dataset_json,
        image_root=config.image_root,
        text_source=config.text_source,
        validate_images=True,
    )
    train_samples = splits.train_samples
    test_images = splits.test_images
    test_texts = splits.test_texts

    if config.debug:
        train_samples = train_samples[: config.debug_train_samples]
        test_images = test_images[: config.debug_eval_images]
        allowed_image_indices = {item["image_index"] for item in test_images}
        test_texts = [item for item in test_texts if item["image_index"] in allowed_image_indices][: config.debug_eval_texts]
        remapped = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_image_indices))}
        for new_idx, item in enumerate(test_images):
            item["image_index"] = new_idx
        for item in test_texts:
            item["image_index"] = remapped[item["image_index"]]

    train_dataset = RSITMDTrainDataset(train_samples, transform=build_train_transform(config.image_size))
    image_dataset = RSITMDImageDataset(test_images, transform=build_eval_transform(config.image_size))
    text_dataset = RSITMDTextDataset(test_texts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=False,
    )
    image_loader = DataLoader(
        image_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )
    text_loader = DataLoader(
        text_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model = build_retrieval_model(config).to(device)
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)

    loss_fn = RetrievalLoss(config.loss_name, sdm_factor=config.sdm_factor, sdm_epsilon=config.sdm_epsilon)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=precision_setup.use_grad_scaler) if device.type == "cuda" else None

    best_metric = float("-inf")
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            precision_setup=precision_setup,
            logger=logger,
            log_interval=config.log_interval,
            grad_clip_norm=config.grad_clip_norm,
            logit_scale_max=config.logit_scale_max,
            scaler=scaler,
        )

        epoch_record: dict[str, float | int | dict] = {"epoch": epoch, "train": train_stats}
        metrics = None
        if epoch % config.eval_interval == 0:
            metrics = evaluate(
                model=model,
                image_loader=image_loader,
                text_loader=text_loader,
                device=device,
                precision_setup=precision_setup,
                chunk_size=config.chunk_size,
                topk=config.topk,
                save_topk=False,
                save_embeddings=False,
            )
            metrics.pop("t2i_ranks", None)
            metrics.pop("i2t_ranks", None)
            logger.info("Validation epoch %d | %s", epoch, format_metrics_for_log(metrics))
            epoch_record["eval"] = metrics

            if metrics["mean_recall"] > best_metric:
                best_metric = float(metrics["mean_recall"])
                best_epoch = epoch
                save_checkpoint(
                    output_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "config": config.to_dict(),
                        "metrics": metrics,
                    },
                )
                save_json(output_dir / "best_metrics.json", {"epoch": best_epoch, "metrics": metrics})
                logger.info("Updated best checkpoint at epoch %d with mean_recall=%.2f", epoch, best_metric)

        history.append(epoch_record)
        save_json(output_dir / "history.json", history)

    logger.info("Training finished. Best epoch=%d | best mean_recall=%.2f", best_epoch, best_metric)


if __name__ == "__main__":
    main()
