from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


VALID_TEXT_SOURCES = {"sentence", "content", "sentence_plus_content"}


def normalize_content_items(content_items: list[str] | None) -> list[str]:
    if not content_items:
        return []
    normalized: list[str] = []
    for item in content_items:
        cleaned = " ".join(str(item).strip().split())
        if cleaned:
            normalized.append(cleaned)
    return normalized


def build_content_text(content_items: list[str] | None) -> str:
    normalized = normalize_content_items(content_items)
    if not normalized:
        return ""
    return ", ".join(normalized)


def build_text(entry: dict[str, Any], sentence_idx: int, text_source: str) -> str:
    if text_source not in VALID_TEXT_SOURCES:
        raise ValueError(f"Unsupported text_source={text_source!r}. Expected one of {sorted(VALID_TEXT_SOURCES)}.")

    sentence = entry["sentences"][sentence_idx]["raw"].strip()
    content_text = build_content_text(entry.get("content"))
    if text_source == "sentence":
        return sentence
    if text_source == "content":
        return content_text or sentence
    if not sentence:
        return content_text
    if not content_text:
        return sentence
    sentence = sentence[:-1] if sentence.endswith(".") else sentence
    return f"{sentence}. scene content: {content_text}"


def resolve_image_root(dataset_json: str, image_root: str = "") -> Path:
    if image_root:
        candidate = Path(image_root)
        if not candidate.exists():
            raise FileNotFoundError(f"Configured image_root does not exist: {candidate}")
        return candidate

    dataset_path = Path(dataset_json).resolve()
    candidates = [
        dataset_path.parent / "images",
        dataset_path.parent / "image",
        dataset_path.parent / "imgs",
        dataset_path.parent / "Images",
        dataset_path.parent.parent / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not auto-detect image_root for {dataset_json}. Tried: {tried}")


def _load_json(dataset_json: str) -> dict[str, Any]:
    with Path(dataset_json).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class RSITMDSplits:
    dataset_name: str
    dataset_json: str
    image_root: Path
    train_samples: list[dict[str, Any]]
    test_images: list[dict[str, Any]]
    test_texts: list[dict[str, Any]]
    stats: dict[str, Any]


def load_rsitmd_splits(
    dataset_json: str,
    image_root: str = "",
    text_source: str = "sentence",
    validate_images: bool = True,
) -> RSITMDSplits:
    payload = _load_json(dataset_json)
    image_root_path = resolve_image_root(dataset_json, image_root=image_root)
    images = payload["images"]

    train_samples: list[dict[str, Any]] = []
    test_images: list[dict[str, Any]] = []
    test_texts: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for entry in images:
        filename = entry["filename"]
        imgid = int(entry["imgid"])
        split = entry["split"]
        image_path = image_root_path / filename
        if validate_images and not image_path.exists():
            missing_images.append(str(image_path))

        if split == "train":
            for sentence_idx, sentence in enumerate(entry["sentences"]):
                train_samples.append(
                    {
                        "image_path": str(image_path),
                        "filename": filename,
                        "imgid": imgid,
                        "sentence_id": int(sentence.get("sentid", sentence_idx)),
                        "sentence_idx": sentence_idx,
                        "raw_text": sentence["raw"].strip(),
                        "text": build_text(entry, sentence_idx, text_source=text_source),
                    }
                )
        elif split == "test":
            image_index = len(test_images)
            test_images.append(
                {
                    "image_path": str(image_path),
                    "filename": filename,
                    "imgid": imgid,
                    "image_index": image_index,
                    "content": normalize_content_items(entry.get("content")),
                }
            )
            for sentence_idx, sentence in enumerate(entry["sentences"]):
                test_texts.append(
                    {
                        "filename": filename,
                        "imgid": imgid,
                        "image_index": image_index,
                        "sentence_id": int(sentence.get("sentid", sentence_idx)),
                        "sentence_idx": sentence_idx,
                        "raw_text": sentence["raw"].strip(),
                        "text": build_text(entry, sentence_idx, text_source=text_source),
                    }
                )
        else:
            raise ValueError(f"Unexpected split value: {split}")

    if missing_images:
        preview = "\n".join(missing_images[:20])
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing image files under {image_root_path}.\nFirst missing paths:\n{preview}"
        )

    stats = {
        "dataset": payload.get("dataset", "RSITMD"),
        "num_images": len(images),
        "num_train_samples": len(train_samples),
        "num_test_images": len(test_images),
        "num_test_texts": len(test_texts),
    }
    return RSITMDSplits(
        dataset_name=payload.get("dataset", "RSITMD"),
        dataset_json=str(Path(dataset_json).resolve()),
        image_root=image_root_path,
        train_samples=train_samples,
        test_images=test_images,
        test_texts=test_texts,
        stats=stats,
    )


class RSITMDTrainDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "text": sample["text"],
            "raw_text": sample["raw_text"],
            "imgid": sample["imgid"],
            "filename": sample["filename"],
            "sentence_id": sample["sentence_id"],
            "sentence_idx": sample["sentence_idx"],
        }


class RSITMDImageDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "imgid": sample["imgid"],
            "filename": sample["filename"],
            "image_index": sample["image_index"],
        }


class RSITMDTextDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "text": sample["text"],
            "raw_text": sample["raw_text"],
            "imgid": sample["imgid"],
            "filename": sample["filename"],
            "sentence_id": sample["sentence_id"],
            "sentence_idx": sample["sentence_idx"],
            "image_index": sample["image_index"],
        }


def build_dataset_sanity_report(
    dataset_json: str,
    image_root: str = "",
    text_source: str = "sentence",
    example_count: int = 3,
) -> dict[str, Any]:
    payload = _load_json(dataset_json)
    image_root_path = resolve_image_root(dataset_json, image_root=image_root)
    split_counts: dict[str, int] = {}
    missing_images: list[str] = []
    examples: list[dict[str, Any]] = []

    for entry in payload["images"]:
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
        image_path = image_root_path / entry["filename"]
        if not image_path.exists():
            missing_images.append(str(image_path))
        if len(examples) < example_count:
            examples.append(
                {
                    "filename": entry["filename"],
                    "split": entry["split"],
                    "sentence": build_text(entry, 0, text_source="sentence"),
                    "content": build_text(entry, 0, text_source="content"),
                    "sentence_plus_content": build_text(entry, 0, text_source="sentence_plus_content"),
                }
            )

    train_caption_count = sum(len(entry["sentences"]) for entry in payload["images"] if entry["split"] == "train")
    test_caption_count = sum(len(entry["sentences"]) for entry in payload["images"] if entry["split"] == "test")
    return {
        "dataset": payload.get("dataset", "RSITMD"),
        "dataset_json": str(Path(dataset_json).resolve()),
        "image_root": str(image_root_path),
        "split_counts": split_counts,
        "train_caption_count": train_caption_count,
        "test_caption_count": test_caption_count,
        "missing_image_count": len(missing_images),
        "missing_images_preview": missing_images[:10],
        "examples": examples,
    }
