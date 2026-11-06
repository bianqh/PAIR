from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .datasets import (
    RSITMDImageDataset,
    RSITMDTextDataset,
    RSITMDTrainDataset,
    build_dataset_sanity_report,
    build_fixed_round_progressive_texts_for_eval,
    load_rsitmd_splits,
    train_collate_fn,
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
    infer_dataset_name,
    resolve_device,
    resolve_precision,
    resolve_dataset_output_base,
    save_checkpoint,
    save_config,
    save_json,
    set_seed,
    setup_logger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CLIP+Qwen retrieval with optional dialogue simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--train-split", default="", help="Override train split name from config.")
    parser.add_argument("--eval-split", default="", help="Override eval split name from config.")
    parser.add_argument("--debug", action="store_true", help="Run a small debug training loop.")
    return parser


def build_optimizer(config, model):
    if str(config.optimizer).lower() != "adamw":
        raise ValueError(f"Unsupported optimizer={config.optimizer!r}. Only 'adamw' is implemented.")

    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim <= 1 or name.endswith(".bias") or "logit_scale" in name:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


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
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

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


def _dialogue_eval_rounds(max_rounds: int) -> list[int]:
    max_rounds = int(max_rounds)
    if max_rounds < 1:
        raise ValueError("eval_progressive_rounds must be >= 1 when use_progressive_dialogue=True.")
    return list(range(1, max_rounds + 1))


def _dialogue_eval_plan(config, has_negation: bool) -> list[tuple[str, int, bool]]:
    rounds = [
        (f"round_{eval_round}", eval_round, False)
        for eval_round in _dialogue_eval_rounds(config.eval_progressive_rounds)
    ]

    if has_negation and int(config.negation_eval_topn) > 0:
        rounds.append((f"round_{config.eval_progressive_rounds + 1}", config.eval_progressive_rounds, True))

    return rounds


def _has_neg_attrs(samples: list[dict]) -> bool:
    return any(bool(sample.get("neg_attrs")) for sample in samples)


def main() -> None:
    args = build_parser().parse_args()

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.train_split:
        overrides["train_split"] = args.train_split
    if args.eval_split:
        overrides["eval_split"] = args.eval_split
    if args.debug:
        overrides["debug"] = True

    config = load_config(args.config, overrides=overrides)

    if config.debug:
        config = apply_debug_settings(config)

    dataset_name = infer_dataset_name(config.dataset_json)
    config.output_dir = str(resolve_dataset_output_base(config.output_dir, dataset_name))

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
    logger.info(
        "Using device=%s | requested_precision=%s | actual_precision=%s",
        device,
        config.precision,
        precision_setup.actual_precision,
    )
    logger.info("%s", precision_setup.reason)

    sanity_report = build_dataset_sanity_report(
        config.dataset_json,
        image_root=config.image_root,
        text_source=config.text_source,
        train_split=config.train_split,
        eval_split=config.eval_split,
    )
    logger.info("Dataset sanity: %s", sanity_report)

    splits = load_rsitmd_splits(
        dataset_json=config.dataset_json,
        image_root=config.image_root,
        text_source=config.text_source,
        validate_images=True,
        use_progressive_dialogue=config.use_progressive_dialogue,
        use_dataset_neg_attrs=config.use_dataset_neg_attrs,
        train_split=config.train_split,
        eval_split=config.eval_split,
    )

    train_samples = splits.train_samples
    eval_images = splits.eval_images
    eval_texts = splits.eval_texts

    logger.info("Loaded splits: %s", splits.stats)

    if config.debug:
        train_samples = train_samples[: config.debug_train_samples]
        eval_images = eval_images[: config.debug_eval_images]

        allowed_image_indices = {item["image_index"] for item in eval_images}
        eval_texts = [item for item in eval_texts if item["image_index"] in allowed_image_indices][: config.debug_eval_texts]

        remapped = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_image_indices))}

        for new_idx, item in enumerate(eval_images):
            item["image_index"] = new_idx

        for item in eval_texts:
            item["image_index"] = remapped[item["image_index"]]

    has_train_neg_attrs = _has_neg_attrs(train_samples)
    has_eval_neg_attrs = _has_neg_attrs(eval_images)

    logger.info(
        "Negation availability | train_has_neg=%s | eval_has_neg=%s | train_neg_samples=%d/%d | eval_neg_images=%d/%d",
        has_train_neg_attrs,
        has_eval_neg_attrs,
        sum(1 for item in train_samples if item.get("neg_attrs")),
        len(train_samples),
        sum(1 for item in eval_images if item.get("neg_attrs")),
        len(eval_images),
    )

    train_dataset = RSITMDTrainDataset(train_samples, transform=build_train_transform(config.image_size))
    image_dataset = RSITMDImageDataset(eval_images, transform=build_eval_transform(config.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        drop_last=False,
        collate_fn=train_collate_fn,
    )

    image_loader = DataLoader(
        image_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model = build_retrieval_model(config).to(device)
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)

    loss_fn = RetrievalLoss(config.loss_name)
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
            use_progressive_dialogue=config.use_progressive_dialogue,
            max_progressive_rounds=config.max_progressive_rounds,
            progressive_dialogue_seed=config.progressive_dialogue_seed,
            negation_train_prob=config.negation_train_prob if has_train_neg_attrs else 0.0,
            negation_train_min_attrs=config.negation_train_min_attrs,
            negation_train_max_attrs=config.negation_train_max_attrs,
        )

        epoch_record: dict[str, float | int | dict] = {"epoch": epoch, "train": train_stats}

        if epoch % config.eval_interval == 0:
            if config.use_progressive_dialogue:
                metrics_by_round: dict[str, dict] = {}

                for round_name, eval_round, include_negation in _dialogue_eval_plan(config, has_eval_neg_attrs):
                    round_texts = build_fixed_round_progressive_texts_for_eval(
                        image_samples=eval_images,
                        eval_round=eval_round,
                        include_negation=include_negation,
                        negation_topn=config.negation_eval_topn,
                    )

                    text_loader = DataLoader(
                        RSITMDTextDataset(round_texts),
                        batch_size=config.eval_batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory and device.type == "cuda",
                    )

                    round_metrics = evaluate(
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

                    round_metrics.pop("t2i_ranks", None)
                    round_metrics.pop("i2t_ranks", None)

                    metrics_by_round[round_name] = round_metrics
                    logger.info("Validation epoch %d %s | %s", epoch, round_name, format_metrics_for_log(round_metrics))

                epoch_record["eval"] = metrics_by_round

                best_round_name = (
                    f"round_{config.eval_progressive_rounds + 1}"
                    if has_eval_neg_attrs and int(config.negation_eval_topn) > 0
                    else f"round_{config.eval_progressive_rounds}"
                )
                metric_for_best = float(metrics_by_round[best_round_name]["mean_recall"])
                metrics_for_checkpoint = metrics_by_round

            else:
                text_loader = DataLoader(
                    RSITMDTextDataset(eval_texts),
                    batch_size=config.eval_batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory and device.type == "cuda",
                )

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
                metric_for_best = float(metrics["mean_recall"])
                metrics_for_checkpoint = metrics

            if metric_for_best > best_metric:
                best_metric = metric_for_best
                best_epoch = epoch

                save_checkpoint(
                    output_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "config": config.to_dict(),
                        "metrics": metrics_for_checkpoint,
                    },
                )
                save_json(output_dir / "best_metrics.json", {"epoch": best_epoch, "metrics": metrics_for_checkpoint})
                logger.info("Updated best checkpoint at epoch %d with mean_recall=%.2f", epoch, best_metric)

        history.append(epoch_record)
        save_json(output_dir / "history.json", history)

    logger.info("Training finished. Best epoch=%d | best mean_recall=%.2f", best_epoch, best_metric)


if __name__ == "__main__":
    main()
