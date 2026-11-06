from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .metrics import compute_retrieval_metrics, compute_topk_matches
from .utils import AverageMeter, get_autocast_context


def _to_device(batch_images: torch.Tensor, batch_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch_images.to(device, non_blocking=True), batch_ids.to(device, non_blocking=True)


def _tokenized_to_device(tokenized_text, device: torch.device):
    if isinstance(tokenized_text, torch.Tensor):
        return tokenized_text.to(device, non_blocking=True)
    if hasattr(tokenized_text, "items"):
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized_text.items()
        }
    if hasattr(tokenized_text, "to"):
        return tokenized_text.to(device)
    raise TypeError(f"Unsupported tokenized text type: {type(tokenized_text).__name__}")


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    epoch: int,
    precision_setup,
    logger,
    log_interval: int,
    grad_clip_norm: float,
    logit_scale_max: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    scale_meter = AverageMeter()

    for step, batch in enumerate(train_loader, start=1):
        images, image_ids = _to_device(batch["image"], batch["imgid"], device)
        tokenized_text = _tokenized_to_device(model.tokenize(list(batch["text"])), device)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = get_autocast_context(precision_setup, device)
        with autocast_context:
            image_features, text_features, logit_scale = model(images, tokenized_text)
            loss = loss_fn(image_features=image_features, text_features=text_features, image_ids=image_ids, logit_scale=logit_scale)

        if scaler is not None and precision_setup.use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        model.clamp_logit_scale(logit_scale_max)
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.detach().float().item())
        scale_value = float(model.get_logit_scale().detach().float().item())
        batch_size = images.size(0)
        loss_meter.update(loss_value, n=batch_size)
        scale_meter.update(scale_value, n=batch_size)

        if step % log_interval == 0 or step == len(train_loader):
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d Step %d/%d | lr=%.8f | loss=%.4f | avg_loss=%.4f | logit_scale=%.4f",
                epoch,
                step,
                len(train_loader),
                current_lr,
                loss_value,
                loss_meter.avg,
                scale_value,
            )

    return {
        "loss": loss_meter.avg,
        "logit_scale": scale_meter.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def encode_image_loader(model, data_loader, device: torch.device, precision_setup) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    autocast_context = get_autocast_context(precision_setup, device)
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context:
            image_features = model.encode_image(images, normalize=True)
        embeddings.append(image_features.float().cpu())
        batch_size = images.size(0)
        for idx in range(batch_size):
            metadata.append(
                {
                    "imgid": int(batch["imgid"][idx]),
                    "filename": batch["filename"][idx],
                    "image_index": int(batch["image_index"][idx]),
                }
            )
    return torch.cat(embeddings, dim=0), metadata


@torch.no_grad()
def encode_text_loader(model, data_loader, device: torch.device, precision_setup) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    autocast_context = get_autocast_context(precision_setup, device)
    for batch in data_loader:
        tokenized_text = _tokenized_to_device(model.tokenize(list(batch["text"])), device)
        with autocast_context:
            text_features = model.encode_text(tokenized_text, normalize=True)
        embeddings.append(text_features.float().cpu())
        batch_size = len(batch["text"])
        for idx in range(batch_size):
            metadata.append(
                {
                    "imgid": int(batch["imgid"][idx]),
                    "filename": batch["filename"][idx],
                    "sentence_id": int(batch["sentence_id"][idx]),
                    "sentence_idx": int(batch["sentence_idx"][idx]),
                    "image_index": int(batch["image_index"][idx]),
                    "text": batch["text"][idx],
                    "raw_text": batch["raw_text"][idx],
                }
            )
    return torch.cat(embeddings, dim=0), metadata


@torch.no_grad()
def evaluate(
    model,
    image_loader,
    text_loader,
    device: torch.device,
    precision_setup,
    chunk_size: int,
    topk: int,
    save_dir: str | Path | None = None,
    save_topk: bool = False,
    save_embeddings: bool = False,
) -> dict[str, Any]:
    image_embeddings, image_meta = encode_image_loader(model, image_loader, device, precision_setup)
    text_embeddings, text_meta = encode_text_loader(model, text_loader, device, precision_setup)

    text_to_image_gt = [item["image_index"] for item in text_meta]
    image_to_text_gt: list[list[int]] = [[] for _ in range(len(image_meta))]
    for text_index, item in enumerate(text_meta):
        image_to_text_gt[item["image_index"]].append(text_index)

    metrics = compute_retrieval_metrics(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        text_to_image_gt=text_to_image_gt,
        image_to_text_gt=image_to_text_gt,
        chunk_size=chunk_size,
    )

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_topk:
            t2i_scores, t2i_indices = compute_topk_matches(text_embeddings, image_embeddings, topk=topk, chunk_size=chunk_size)
            i2t_scores, i2t_indices = compute_topk_matches(image_embeddings, text_embeddings, topk=topk, chunk_size=chunk_size)
            topk_payload = {
                "text_to_image": [
                    {
                        "query_index": idx,
                        "text": meta["text"],
                        "gt_filename": image_meta[meta["image_index"]]["filename"],
                        "gt_imgid": image_meta[meta["image_index"]]["imgid"],
                        "topk": [
                            {
                                "rank": rank + 1,
                                "filename": image_meta[int(candidate_index)]["filename"],
                                "imgid": image_meta[int(candidate_index)]["imgid"],
                                "score": float(score),
                                "is_gt": int(candidate_index) == meta["image_index"],
                            }
                            for rank, (score, candidate_index) in enumerate(zip(t2i_scores[idx].tolist(), t2i_indices[idx].tolist()))
                        ],
                    }
                    for idx, meta in enumerate(text_meta)
                ],
                "image_to_text": [
                    {
                        "query_index": idx,
                        "filename": meta["filename"],
                        "imgid": meta["imgid"],
                        "gt_sentence_ids": [text_meta[text_idx]["sentence_id"] for text_idx in image_to_text_gt[idx]],
                        "topk": [
                            {
                                "rank": rank + 1,
                                "sentence_id": text_meta[int(candidate_index)]["sentence_id"],
                                "text": text_meta[int(candidate_index)]["text"],
                                "score": float(score),
                                "is_gt": int(candidate_index) in image_to_text_gt[idx],
                            }
                            for rank, (score, candidate_index) in enumerate(zip(i2t_scores[idx].tolist(), i2t_indices[idx].tolist()))
                        ],
                    }
                    for idx, meta in enumerate(image_meta)
                ],
            }
            from .utils import save_json

            save_json(output_dir / "topk_results.json", topk_payload)

        if save_embeddings:
            torch.save(
                {
                    "image_embeddings": image_embeddings,
                    "text_embeddings": text_embeddings,
                    "image_meta": image_meta,
                    "text_meta": text_meta,
                },
                output_dir / "embeddings.pt",
            )

    metrics["image_count"] = len(image_meta)
    metrics["text_count"] = len(text_meta)
    return metrics
