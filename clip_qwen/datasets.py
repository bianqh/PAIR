from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_TEXT_SOURCES = {"sentence", "content", "sentence_plus_content"}
NEGATION_TEMPLATE = "There is no {neg_attr_list} in this scene."
DIALOGUE_QUESTION_TEMPLATES = [
    "Briefly describe the overall scene you are looking for?",
    "What are the main objects or regions in the scene?",
    "How are these objects spatially arranged?",
    "Are there any additional details that help identify the scene?",
]
DIALOGUE_EXCLUSION_TEMPLATE = (
    "What is absent from your target scene that may appear in similar scenes?"
)


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


def normalize_neg_items(neg_items: list[str] | None) -> list[str]:
    if not neg_items:
        return []

    normalized: list[str] = []
    seen: set[str] = set()

    for item in neg_items:
        if isinstance(item, dict):
            item = item.get("attr", item.get("phrase", item.get("neg_attr", "")))
        cleaned = " ".join(str(item).strip().split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)

    return normalized


def extract_neg_attrs(entry: dict[str, Any], use_dataset_neg_attrs: bool = True) -> list[str]:
    if not use_dataset_neg_attrs:
        return []

    if isinstance(entry.get("hard_negative_content"), list):
        return normalize_neg_items(entry.get("hard_negative_content"))

    if isinstance(entry.get("neg_attrs"), list):
        return normalize_neg_items(entry.get("neg_attrs"))

    return []


def _format_neg_attr_list(neg_attrs: list[str]) -> str:
    attrs = [" ".join(str(item).strip().split()) for item in neg_attrs if str(item).strip()]
    if not attrs:
        return ""
    if len(attrs) == 1:
        return attrs[0]
    if len(attrs) == 2:
        return f"{attrs[0]} or {attrs[1]}"
    return f"{', '.join(attrs[:-1])}, or {attrs[-1]}"


def _build_negation_sentence(neg_attrs: str | list[str] | None) -> str:
    if isinstance(neg_attrs, str) or neg_attrs is None:
        attrs = [neg_attrs] if neg_attrs else []
    else:
        attrs = list(neg_attrs)

    attr_list = _format_neg_attr_list(attrs)
    if not attr_list:
        return ""

    return NEGATION_TEMPLATE.format(neg_attr_list=attr_list)


def append_negation_sentence(
    text: str,
    neg_attrs: str | list[str] | None,
) -> str:
    negation = _build_negation_sentence(neg_attrs)
    if not negation:
        return text

    return f"{text} {DIALOGUE_EXCLUSION_TEMPLATE} {negation}".strip()


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


def _ensure_sentence_terminal(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return cleaned + "."


def normalize_progressive_rounds(round_items: list[str] | None) -> list[str]:
    if not round_items:
        return []

    normalized: list[str] = []

    for item in round_items:
        cleaned = _ensure_sentence_terminal(item)
        if cleaned:
            normalized.append(cleaned)

    return normalized


def extract_progressive_rounds(entry: dict[str, Any]) -> list[str]:
    if isinstance(entry.get("progressive_rounds"), list):
        return normalize_progressive_rounds(entry.get("progressive_rounds"))

    ordered = [
        entry.get("round1", ""),
        entry.get("round2", ""),
        entry.get("round3", ""),
        entry.get("round4", ""),
    ]
    return normalize_progressive_rounds(ordered)


def build_progressive_prefix_text(round_items: list[str], round_idx: int | None = None) -> str:
    cleaned = normalize_progressive_rounds(round_items)
    if not cleaned:
        return ""

    if round_idx is None:
        round_idx = len(cleaned)

    round_idx = min(max(int(round_idx), 1), len(cleaned))
    return " ".join(cleaned[:round_idx])


def build_dialogue_question_prefix_text(round_items: list[str], round_idx: int | None = None) -> str:
    cleaned = normalize_progressive_rounds(round_items)
    if not cleaned:
        return ""

    if round_idx is None:
        round_idx = len(cleaned)

    round_idx = min(max(int(round_idx), 1), len(cleaned), len(DIALOGUE_QUESTION_TEMPLATES))
    turns: list[str] = []
    for index, answer in enumerate(cleaned[:round_idx]):
        question = DIALOGUE_QUESTION_TEMPLATES[index]
        turns.append(f"{question} {answer}")
    return " ".join(turns)


def build_progressive_query_text(
    round_items: list[str],
    round_idx: int | None = None,
) -> str:
    return build_dialogue_question_prefix_text(round_items, round_idx=round_idx)


def _max_available_progressive_round(round_count: int) -> int:
    return max(int(round_count), 0)


def sample_progressive_dialogue_text(
    round_items: list[str],
    seed: int,
    round_idx: int | None = None,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    cleaned = normalize_progressive_rounds(round_items)
    if not cleaned:
        return {"text": "", "used_rounds": [], "round_idx": 0}

    max_available_round = _max_available_progressive_round(len(cleaned))
    if max_rounds is not None and max_rounds > 0:
        max_available_round = min(max_available_round, int(max_rounds))
    max_available_round = max(max_available_round, 1)

    rng = random.Random(seed)
    if round_idx is None:
        sampled_round = rng.randint(1, max_available_round)
    else:
        sampled_round = min(max(int(round_idx), 1), max_available_round)

    used_rounds = cleaned[:sampled_round]

    return {
        "text": build_progressive_query_text(
            cleaned,
            round_idx=sampled_round,
        ),
        "used_rounds": used_rounds,
        "round_idx": sampled_round,
    }


def build_fixed_round_progressive_texts_for_eval(
    image_samples: list[dict[str, Any]],
    eval_round: int,
    include_negation: bool = False,
    negation_topn: int = 1,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []

    for sample in image_samples:
        rounds = normalize_progressive_rounds(sample.get("progressive_rounds"))
        if not rounds:
            continue

        round_idx = min(max(int(eval_round), 1), len(rounds))
        text = build_progressive_query_text(
            rounds,
            round_idx=round_idx,
        )

        if include_negation:
            topn = max(int(negation_topn), 0)
            text = append_negation_sentence(
                text,
                sample.get("neg_attrs", [])[:topn],
            )

        if not text:
            continue

        augmented.append(
            {
                "filename": sample["filename"],
                "imgid": sample["imgid"],
                "image_index": sample["image_index"],
                "sentence_id": sample["imgid"],
                "sentence_idx": 0,
                "raw_text": text,
                "text": text,
                "progressive_rounds": rounds,
                "neg_attrs": sample.get("neg_attrs", []),
            }
        )

    return augmented


@dataclass
class RSITMDSplits:
    dataset_name: str
    dataset_json: str
    image_root: Path
    train_split: str
    eval_split: str
    train_samples: list[dict[str, Any]]
    eval_images: list[dict[str, Any]]
    eval_texts: list[dict[str, Any]]
    stats: dict[str, Any]

    @property
    def test_images(self) -> list[dict[str, Any]]:
        return self.eval_images

    @property
    def test_texts(self) -> list[dict[str, Any]]:
        return self.eval_texts


def load_rsitmd_splits(
    dataset_json: str,
    image_root: str = "",
    text_source: str = "sentence",
    validate_images: bool = True,
    use_progressive_dialogue: bool = False,
    use_dataset_neg_attrs: bool = True,
    train_split: str = "train",
    eval_split: str = "test",
) -> RSITMDSplits:
    payload = _load_json(dataset_json)
    image_root_path = resolve_image_root(dataset_json, image_root=image_root)
    images = payload["images"]
    train_split_key = str(train_split).strip().lower()
    eval_split_key = str(eval_split).strip().lower()

    if not train_split_key or not eval_split_key:
        raise ValueError("train_split and eval_split must be non-empty strings.")
    if train_split_key == eval_split_key:
        raise ValueError("train_split and eval_split must be different.")

    train_samples: list[dict[str, Any]] = []
    eval_images: list[dict[str, Any]] = []
    eval_texts: list[dict[str, Any]] = []

    missing_images: list[str] = []
    missing_progressive: list[str] = []
    split_counts: dict[str, int] = {}
    ignored_split_counts: dict[str, int] = {}

    for entry in images:
        filename = entry["filename"]
        imgid = int(entry["imgid"])
        split = str(entry["split"]).strip().lower()
        split_counts[split] = split_counts.get(split, 0) + 1
        image_path = image_root_path / filename

        if validate_images and not image_path.exists():
            missing_images.append(str(image_path))

        progressive_rounds = extract_progressive_rounds(entry)
        neg_attrs = extract_neg_attrs(entry, use_dataset_neg_attrs=use_dataset_neg_attrs)

        if split == train_split_key:
            if use_progressive_dialogue:
                if not progressive_rounds:
                    missing_progressive.append(filename)
                    continue

                train_samples.append(
                    {
                        "image_path": str(image_path),
                        "filename": filename,
                        "imgid": imgid,
                        "sentence_id": imgid,
                        "sentence_idx": 0,
                        "raw_text": "",
                        "text": build_progressive_query_text(progressive_rounds),
                        "progressive_rounds": progressive_rounds,
                        "neg_attrs": neg_attrs,
                    }
                )
                continue

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
                        "progressive_rounds": progressive_rounds,
                        "neg_attrs": neg_attrs,
                    }
                )

        elif split == eval_split_key:
            if use_progressive_dialogue and not progressive_rounds:
                missing_progressive.append(filename)
                continue

            image_index = len(eval_images)
            eval_images.append(
                {
                    "image_path": str(image_path),
                    "filename": filename,
                    "imgid": imgid,
                    "image_index": image_index,
                    "progressive_rounds": progressive_rounds,
                    "neg_attrs": neg_attrs,
                }
            )

            if use_progressive_dialogue:
                eval_texts.append(
                    {
                        "filename": filename,
                        "imgid": imgid,
                        "image_index": image_index,
                        "sentence_id": imgid,
                        "sentence_idx": 0,
                        "raw_text": "",
                        "text": build_progressive_query_text(progressive_rounds),
                        "progressive_rounds": progressive_rounds,
                        "neg_attrs": neg_attrs,
                    }
                )
            else:
                for sentence_idx, sentence in enumerate(entry["sentences"]):
                    eval_texts.append(
                        {
                            "filename": filename,
                            "imgid": imgid,
                            "image_index": image_index,
                            "sentence_id": int(sentence.get("sentid", sentence_idx)),
                            "sentence_idx": sentence_idx,
                            "raw_text": sentence["raw"].strip(),
                            "text": build_text(entry, sentence_idx, text_source=text_source),
                            "progressive_rounds": progressive_rounds,
                            "neg_attrs": neg_attrs,
                        }
                    )

        else:
            ignored_split_counts[split] = ignored_split_counts.get(split, 0) + 1

    if missing_images:
        preview = "\n".join(missing_images[:20])
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing image files under {image_root_path}.\nFirst missing paths:\n{preview}"
        )

    if train_split_key not in split_counts:
        available = ", ".join(sorted(split_counts))
        raise ValueError(f"train_split={train_split!r} not found in dataset splits. Available splits: {available}")

    if eval_split_key not in split_counts:
        available = ", ".join(sorted(split_counts))
        raise ValueError(f"eval_split={eval_split!r} not found in dataset splits. Available splits: {available}")

    if use_progressive_dialogue and missing_progressive:
        preview = ", ".join(missing_progressive[:20])
        raise ValueError(
            "use_progressive_dialogue=True but some entries do not contain valid round1~round4 "
            f"or progressive_rounds. First examples: {preview}"
        )

    train_neg_count = sum(1 for sample in train_samples if sample.get("neg_attrs"))
    eval_image_neg_count = sum(1 for sample in eval_images if sample.get("neg_attrs"))
    eval_text_neg_count = sum(1 for sample in eval_texts if sample.get("neg_attrs"))

    stats = {
        "dataset": payload.get("dataset", "RSITMD"),
        "num_images": len(images),
        "split_counts": split_counts,
        "train_split": train_split_key,
        "eval_split": eval_split_key,
        "ignored_split_counts": ignored_split_counts,
        "num_train_samples": len(train_samples),
        "num_eval_images": len(eval_images),
        "num_eval_texts": len(eval_texts),
        "use_progressive_dialogue": use_progressive_dialogue,
        "use_dataset_neg_attrs": use_dataset_neg_attrs,
        "num_train_samples_with_neg_attrs": train_neg_count,
        "num_eval_images_with_neg_attrs": eval_image_neg_count,
        "num_eval_texts_with_neg_attrs": eval_text_neg_count,
    }

    return RSITMDSplits(
        dataset_name=payload.get("dataset", "RSITMD"),
        dataset_json=str(Path(dataset_json).resolve()),
        image_root=image_root_path,
        train_split=train_split_key,
        eval_split=eval_split_key,
        train_samples=train_samples,
        eval_images=eval_images,
        eval_texts=eval_texts,
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
            "progressive_rounds": sample.get("progressive_rounds", []),
            "neg_attrs": sample.get("neg_attrs", []),
            "sample_index": index,
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
            "progressive_rounds": sample.get("progressive_rounds", []),
            "neg_attrs": sample.get("neg_attrs", []),
        }


def train_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "text": [item["text"] for item in batch],
        "raw_text": [item["raw_text"] for item in batch],
        "imgid": torch.tensor([int(item["imgid"]) for item in batch], dtype=torch.long),
        "filename": [item["filename"] for item in batch],
        "sentence_id": torch.tensor([int(item["sentence_id"]) for item in batch], dtype=torch.long),
        "sentence_idx": torch.tensor([int(item["sentence_idx"]) for item in batch], dtype=torch.long),
        "progressive_rounds": [item.get("progressive_rounds", []) for item in batch],
        "neg_attrs": [item.get("neg_attrs", []) for item in batch],
        "sample_index": torch.tensor([int(item["sample_index"]) for item in batch], dtype=torch.long),
    }


def build_dataset_sanity_report(
    dataset_json: str,
    image_root: str = "",
    text_source: str = "sentence",
    example_count: int = 3,
    train_split: str = "train",
    eval_split: str = "test",
) -> dict[str, Any]:
    payload = _load_json(dataset_json)
    image_root_path = resolve_image_root(dataset_json, image_root=image_root)
    train_split_key = str(train_split).strip().lower()
    eval_split_key = str(eval_split).strip().lower()

    split_counts: dict[str, int] = {}
    caption_counts_by_split: dict[str, int] = {}
    missing_images: list[str] = []
    examples: list[dict[str, Any]] = []

    hard_negative_entry_count = 0
    hard_negative_total = 0
    progressive_count = 0

    for entry in payload["images"]:
        split = str(entry["split"]).strip().lower()
        split_counts[split] = split_counts.get(split, 0) + 1
        caption_counts_by_split[split] = caption_counts_by_split.get(split, 0) + len(entry["sentences"])

        image_path = image_root_path / entry["filename"]
        if not image_path.exists():
            missing_images.append(str(image_path))

        progressive_rounds = extract_progressive_rounds(entry)
        if progressive_rounds:
            progressive_count += 1

        neg_attrs = extract_neg_attrs(entry, use_dataset_neg_attrs=True)
        if neg_attrs:
            hard_negative_entry_count += 1
            hard_negative_total += len(neg_attrs)

        if len(examples) < example_count:
            examples.append(
                {
                    "filename": entry["filename"],
                    "split": entry["split"],
                    "sentence": build_text(entry, 0, text_source="sentence"),
                    "content": build_text(entry, 0, text_source="content"),
                    "sentence_plus_content": build_text(entry, 0, text_source="sentence_plus_content"),
                    "round1": entry.get("round1", ""),
                    "round2": entry.get("round2", ""),
                    "round3": entry.get("round3", ""),
                    "round4": entry.get("round4", ""),
                    "hard_negative_content": neg_attrs,
                }
            )

    train_caption_count = caption_counts_by_split.get(train_split_key, 0)
    eval_caption_count = caption_counts_by_split.get(eval_split_key, 0)

    return {
        "dataset": payload.get("dataset", "RSITMD"),
        "dataset_json": str(Path(dataset_json).resolve()),
        "image_root": str(image_root_path),
        "split_counts": split_counts,
        "train_split": train_split_key,
        "eval_split": eval_split_key,
        "caption_counts_by_split": caption_counts_by_split,
        "train_caption_count": train_caption_count,
        "eval_caption_count": eval_caption_count,
        "progressive_entry_count": progressive_count,
        "hard_negative_entry_count": hard_negative_entry_count,
        "avg_hard_negative_len": hard_negative_total / max(hard_negative_entry_count, 1),
        "missing_image_count": len(missing_images),
        "missing_images_preview": missing_images[:10],
        "examples": examples,
    }
