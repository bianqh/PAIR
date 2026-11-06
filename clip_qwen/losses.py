from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_itc_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64, device=image_features.device)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


class RetrievalLoss(nn.Module):
    def __init__(self, loss_name: str = "itc") -> None:
        super().__init__()
        self.loss_name = loss_name

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_ids: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        _ = image_ids
        if self.loss_name == "itc":
            return compute_itc_loss(image_features=image_features, text_features=text_features, logit_scale=logit_scale)
        raise ValueError(f"Unsupported loss_name={self.loss_name!r}. Expected 'itc'.")
