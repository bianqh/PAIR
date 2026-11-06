from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _ensure_fp32_normalized(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = embeddings.float()
    return torch.nn.functional.normalize(embeddings, dim=-1)


def _compute_ranks_single_positive(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gt_indices: list[int],
    chunk_size: int,
) -> list[int]:
    gallery_t = gallery_embeddings.t()
    ranks: list[int] = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start : start + chunk_size]
        scores = chunk @ gallery_t
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(scores.shape[0]):
            gt_index = gt_indices[start + row_idx]
            rank = int((sorted_indices[row_idx] == gt_index).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
    return ranks


def _compute_ranks_multi_positive(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gt_indices: list[list[int]],
    chunk_size: int,
) -> list[int]:
    gallery_t = gallery_embeddings.t()
    ranks: list[int] = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start : start + chunk_size]
        scores = chunk @ gallery_t
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(scores.shape[0]):
            positives = gt_indices[start + row_idx]
            positive_mask = torch.isin(sorted_indices[row_idx], torch.tensor(positives, device=sorted_indices.device))
            rank = int(positive_mask.nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
    return ranks


def _summarize_ranks(ranks: list[int]) -> dict[str, float]:
    ranks_np = np.asarray(ranks, dtype=np.int64)
    metrics = {
        "R@1": float((ranks_np <= 1).mean() * 100.0),
        "R@5": float((ranks_np <= 5).mean() * 100.0),
        "R@10": float((ranks_np <= 10).mean() * 100.0),
        "MedR": float(np.median(ranks_np)),
        "MeanR": float(ranks_np.mean()),
    }
    metrics["mean_recall"] = float((metrics["R@1"] + metrics["R@5"] + metrics["R@10"]) / 3.0)
    return metrics


def compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_to_image_gt: list[int],
    image_to_text_gt: list[list[int]],
    chunk_size: int = 1024,
) -> dict[str, Any]:
    image_embeddings = _ensure_fp32_normalized(image_embeddings.cpu())
    text_embeddings = _ensure_fp32_normalized(text_embeddings.cpu())

    t2i_ranks = _compute_ranks_single_positive(
        query_embeddings=text_embeddings,
        gallery_embeddings=image_embeddings,
        gt_indices=text_to_image_gt,
        chunk_size=chunk_size,
    )
    i2t_ranks = _compute_ranks_multi_positive(
        query_embeddings=image_embeddings,
        gallery_embeddings=text_embeddings,
        gt_indices=image_to_text_gt,
        chunk_size=chunk_size,
    )

    t2i_metrics = _summarize_ranks(t2i_ranks)
    i2t_metrics = _summarize_ranks(i2t_ranks)
    mean_recall = float((t2i_metrics["mean_recall"] + i2t_metrics["mean_recall"]) / 2.0)
    return {
        "t2i": t2i_metrics,
        "i2t": i2t_metrics,
        "mean_recall": mean_recall,
        "t2i_ranks": t2i_ranks,
        "i2t_ranks": i2t_ranks,
    }


def compute_topk_matches(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    topk: int,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_embeddings = _ensure_fp32_normalized(query_embeddings.cpu())
    gallery_embeddings = _ensure_fp32_normalized(gallery_embeddings.cpu())
    gallery_t = gallery_embeddings.t()
    topk_scores: list[torch.Tensor] = []
    topk_indices: list[torch.Tensor] = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start : start + chunk_size]
        scores = chunk @ gallery_t
        chunk_scores, chunk_indices = torch.topk(scores, k=min(topk, scores.shape[1]), dim=1)
        topk_scores.append(chunk_scores)
        topk_indices.append(chunk_indices)
    return torch.cat(topk_scores, dim=0), torch.cat(topk_indices, dim=0)
