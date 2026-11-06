from __future__ import annotations

import random

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class RandomDiscreteRotation:
    def __init__(self, angles: tuple[int, ...] = (0, 90, 180, 270)) -> None:
        self.angles = angles

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        if angle == 0:
            return image
        return image.rotate(angle)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomDiscreteRotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ]
    )
