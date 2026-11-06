from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_sdm_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_ids: torch.Tensor,
    logit_scale: torch.Tensor,
    image_id: torch.Tensor | None = None,
    factor: float = 0.3,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Similarity Distribution Matching.

    This follows the existing `demo/loss.py::compute_sdm` behavior on purpose
    so the baseline stays numerically aligned with the current repository.
    """
    batch_size = image_features.shape[0]
    image_ids = image_ids.reshape((batch_size, 1))
    pid_dist = image_ids - image_ids.t()
    labels = (pid_dist == 0).float()

    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    return torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))


def compute_itc_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64, device=image_features.device)

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


class RetrievalLoss(nn.Module):
    def __init__(self, loss_name: str = "sdm", sdm_factor: float = 0.3, sdm_epsilon: float = 1e-8) -> None:
        super().__init__()
        self.loss_name = loss_name
        self.sdm_factor = sdm_factor
        self.sdm_epsilon = sdm_epsilon

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_ids: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_name == "sdm":
            return compute_sdm_loss(
                image_features=image_features,
                text_features=text_features,
                image_ids=image_ids,
                logit_scale=logit_scale,
                factor=self.sdm_factor,
                epsilon=self.sdm_epsilon,
            )
        if self.loss_name == "itc":
            return compute_itc_loss(image_features=image_features, text_features=text_features, logit_scale=logit_scale)
        raise ValueError(f"Unsupported loss_name={self.loss_name!r}. Expected 'sdm' or 'itc'.")
