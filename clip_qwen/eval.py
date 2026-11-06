from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from .config import load_config
from .datasets import (
    RSITMDImageDataset,
    RSITMDTextDataset,
    build_dataset_sanity_report,
    build_fixed_round_progressive_texts_for_eval,
    load_rsitmd_splits,
)
from .engine import evaluate
from .model import build_retrieval_model
from .transforms import build_eval_transform
from .utils import (
    configure_tf32,
    find_latest_timestamped_output_dir,
    format_metrics_for_log,
    get_timestamp_string,
    infer_dataset_name,
    load_checkpoint,
    resolve_device,
    resolve_precision,
    resolve_dataset_output_base,
    save_config,
    save_json,
    set_seed,
    setup_logger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CLIP+Qwen progressive retrieval.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to evaluate. If empty, evaluate pretrained-only.")
    parser.add_argument("--output-dir", default="", help="Override evaluation output directory.")
    parser.add_argument("--train-split", default="", help="Override train split name from config.")
    parser.add_argument("--eval-split", default="", help="Override eval split name from config.")
    parser.add_argument("--debug", action="store_true", help="Run evaluation on a small subset.")
    return parser


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
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.train_split:
        overrides["train_split"] = args.train_split
    if args.eval_split:
        overrides["eval_split"] = args.eval_split
    if args.debug:
        overrides["debug"] = True

    config = load_config(args.config, overrides=overrides)
    dataset_name = infer_dataset_name(config.dataset_json)
    config.output_dir = str(resolve_dataset_output_base(config.output_dir, dataset_name))

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

    eval_images = splits.eval_images
    eval_texts = splits.eval_texts

    logger.info("Loaded splits: %s", splits.stats)

    if config.debug:
        eval_images = eval_images[: config.debug_eval_images]

        allowed_image_indices = {item["image_index"] for item in eval_images}
        eval_texts = [item for item in eval_texts if item["image_index"] in allowed_image_indices][: config.debug_eval_texts]

        remapped = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_image_indices))}

        for new_idx, item in enumerate(eval_images):
            item["image_index"] = new_idx

        for item in eval_texts:
            item["image_index"] = remapped[item["image_index"]]

    has_eval_neg_attrs = _has_neg_attrs(eval_images)

    logger.info(
        "Negation availability | eval_has_neg=%s | eval_neg_images=%d/%d",
        has_eval_neg_attrs,
        sum(1 for item in eval_images if item.get("neg_attrs")),
        len(eval_images),
    )

    image_dataset = RSITMDImageDataset(eval_images, transform=build_eval_transform(config.image_size))
    image_loader = DataLoader(
        image_dataset,
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

            round_dir = output_dir / round_name

            metrics = evaluate(
                model=model,
                image_loader=image_loader,
                text_loader=text_loader,
                device=device,
                precision_setup=precision_setup,
                chunk_size=config.chunk_size,
                topk=config.topk,
                save_dir=round_dir,
                save_topk=config.save_topk,
                save_embeddings=config.save_embeddings,
            )

            metrics_by_round[round_name] = metrics
            logger.info("Evaluation %s finished | %s", round_name, format_metrics_for_log(metrics))

        save_json(output_dir / "metrics.json", metrics_by_round)

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
            save_dir=output_dir,
            save_topk=config.save_topk,
            save_embeddings=config.save_embeddings,
        )

        save_json(output_dir / "metrics.json", metrics)
        logger.info("Evaluation finished | %s", format_metrics_for_log(metrics))


if __name__ == "__main__":
    main()
