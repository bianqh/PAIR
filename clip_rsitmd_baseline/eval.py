from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from .config import load_config
from .datasets import RSITMDImageDataset, RSITMDTextDataset, build_dataset_sanity_report, load_rsitmd_splits
from .engine import evaluate
from .model import build_retrieval_model
from .transforms import build_eval_transform
from .utils import (
    configure_tf32,
    find_latest_timestamped_output_dir,
    format_metrics_for_log,
    get_timestamp_string,
    load_checkpoint,
    resolve_device,
    resolve_precision,
    save_config,
    save_json,
    set_seed,
    setup_logger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RSITMD CLIP baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to evaluate. If empty, evaluate pretrained-only.")
    parser.add_argument("--output-dir", default="", help="Override evaluation output directory.")
    parser.add_argument("--debug", action="store_true", help="Run evaluation on a small subset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = {}
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.debug:
        overrides["debug"] = True

    config = load_config(args.config, overrides=overrides)
    if args.output_dir:
        output_root = Path(config.output_dir)
    elif config.checkpoint:
        output_root = Path(config.checkpoint).resolve().parent
    else:
        output_root = find_latest_timestamped_output_dir(config.output_dir) or Path(config.output_dir)

    output_dir = output_root / ("eval_debug" if config.debug else "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, f"eval_{get_timestamp_string()}.log")
    save_config(config, output_dir)

    device = resolve_device(config.device)
    precision_setup = resolve_precision(config.amp, config.precision, device)
    configure_tf32(config.allow_tf32)
    set_seed(config.seed)

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
    test_images = splits.test_images
    test_texts = splits.test_texts

    if config.debug:
        test_images = test_images[: config.debug_eval_images]
        allowed_image_indices = {item["image_index"] for item in test_images}
        test_texts = [item for item in test_texts if item["image_index"] in allowed_image_indices][: config.debug_eval_texts]
        remapped = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_image_indices))}
        for new_idx, item in enumerate(test_images):
            item["image_index"] = new_idx
        for item in test_texts:
            item["image_index"] = remapped[item["image_index"]]

    image_dataset = RSITMDImageDataset(test_images, transform=build_eval_transform(config.image_size))
    text_dataset = RSITMDTextDataset(test_texts)

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

    if config.checkpoint:
        load_checkpoint(config.checkpoint, model, strict=True)
        logger.info("Loaded fine-tuned checkpoint: %s", config.checkpoint)
    else:
        logger.info("No checkpoint provided. Evaluating pretrained-only model.")

    metrics = evaluate(
        model=model,
        image_loader=image_loader,
        text_loader=text_loader,
        device=device,
        precision_setup=precision_setup,
        chunk_size=config.chunk_size,
        topk=config.topk,
        save_dir=output_dir,
        save_topk=config.save_topk,
        save_embeddings=config.save_embeddings,
    )
    save_json(output_dir / "metrics.json", metrics)
    logger.info("Evaluation finished | %s", format_metrics_for_log(metrics))


if __name__ == "__main__":
    main()
