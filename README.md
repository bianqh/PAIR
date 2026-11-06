# CLIP + Qwen Remote Sensing Image-Text Retrieval Baseline — Handover Documentation

This package contains the remote sensing image-text retrieval baseline code for "CLIP Vision Tower + Qwen3-0.6B Text Tower",  packaged along with the **reconstructed dataset JSON**. After unzipping, complete the data and weights to start fine-tuning and evaluation.

## Method Overview

- **Vision Tower**: OpenCLIP `ViT-B-32`, initialized with RS5M pre-trained weights.
- **Text Tower**: Qwen3-0.6B, initialized with `best_student.pt` obtained from "RS5M CLIP Text Space Distillation".
- **Training Objective**: ITC contrastive loss (image-text bidirectional InfoNCE).
- **Data Protocol**: Progressive four-round dialogue-style query (`round1` to `round4`);  optional negative attribute augmentation (`hard_negative_content`, training appends negative sentences with probability `negation_train_prob`,  evaluation additionally evaluates `round_5`).
- A pure CLIP control baseline `clip_rsitmd_baseline` is also included in the package (optional, can be ignored if not run).

## 1. Directory Structure

```text
qwen_retrieval_baseline/
├── README_EN.md          ← This file
├── requirements.txt
├── train.py / eval.py           ← Root entry points (automatically dispatch to sub-packages based on config path)
├── clip_qwen/                   ← Main method: CLIP + Qwen retrieval training/evaluation
│   └── configs/
│       ├── rsitmd_clip_qwen_itc.yaml
│       └── rsicd_clip_qwen_itc.yaml
├── qwen_text_distill/           ← Text tower distillation (for reproducing `best_student.pt`, optional)
├── clip_rsitmd_baseline/        ← Pure CLIP control baseline (optional)
└── datasets/
    ├── RSITMD/dataset_RSITMD.json      (4743 images: train 4291 / test 452)
    └── RSICD/dataset_rsicd.json        (10921 images: train 8734 / val 1094 / test 1093)
```

> All paths in configurations (`datasets/...`, `weights/...`) are relative paths. 
> **Please execute commands in the package root directory** (i.e., under `qwen_retrieval_baseline/`).

## 2. Three Things That Need to Be Completed

### (a) Original Images

| Dataset | Count | Format | Destination Path |
|---|---|---|---|
| RSITMD | 4743 | `.tif` | `datasets/RSITMD/images/` |
| RSICD | 10921 | `.jpg` (e.g., `00001.jpg`) | `datasets/RSICD/RSICD_images/` |

Only file names are stored in the JSON, excluding the images themselves. If the directory name is inconsistent with the above,  modify `image_root` in the config (when this field is empty for RSITMD, it automatically looks for `images/` in the same directory as the JSON).

### (b) Qwen3-0.6B Base Model

Download from HuggingFace to `weights/`:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir weights
```

`weights/` requires at least: `config.json`, `tokenizer.json`, `merges.txt`,  `vocab.json`, `tokenizer_config.json`.

### (c) Initialization Weights (Missing will cause direct errors/performance drops)

| File | Size | Purpose | Consequence of Absence |
|---|---|---|---|
| `weights/RS5M_ViT-B-32_RET-2.pt` | ~605 MB | RS5M pre-trained weights for the CLIP vision tower | **Direct `FileNotFoundError`, unable to run** |
| `weights/best_student.pt` | ~1.2 GB | Qwen text tower distillation weights (includes `qwen.*` and `text_projection.*`) | Can run, but the text tower lacks CLIP alignment initialization, resulting in significant performance drop |

Notes:

- When `best_student.pt` is present, the code **will not** read Qwen base model weights, but only reads  `config.json` + tokenizer from `weights/`, so it can run even without large `*.safetensors` files in (b).
- If `best_student.pt` is temporarily unavailable: change `qwen_weight_path: weights/best_student.pt`  in the config to `qwen_weight_path: ""`. In this case, complete Qwen3-0.6B base model weights ((b) full download) must be provided.
- `qwen_text_distill/` is the code to reproduce `best_student.pt` (Teacher = RS5M CLIP text tower,  Student = Qwen text tower, Loss = 1-cos). The Pub11 metadata CSV it uses is public and can be downloaded manually (approx. 1 GB):
  
  ```bash
  # Place under datasets/, keep the filename as pub11_*_metadata.csv
  curl -L -o datasets/pub11_train_metadata.csv \
    https://huggingface.co/datasets/Zilun/RS5M/resolve/main/pub11_train_metadata.csv
  curl -L -o datasets/pub11_validation_metadata.csv \
    https://huggingface.co/datasets/Zilun/RS5M/resolve/main/pub11_validation_metadata.csv
  python -m qwen_text_distill.train_distill --config qwen_text_distill/configs/rs5m_qwen_text_distill.yaml
  ```
  
  Note: Both `weights/RS5M_ViT-B-32_RET-2.pt` and `weights/best_student.pt` are already included in the project folder in the versions used in this paper; please use these included files directly for paper reproduction. If you retrain `best_student.pt` on your machine, the resulting weights may not be completely consistent with the included paper version. In addition, a similar OpenCLIP ViT-B-32 checkpoint for RS5M CLIP weights is publicly available on HuggingFace (`WineDuck/openclip-rs5m`), but it is not the same file as `RS5M_ViT-B-32_RET-2.pt`; for exact reproduction, please use the version included in this project.

## 3. Environment Installation

```bash
pip install -r requirements.txt
```

Please install `torch`/`torchvision` from pytorch.org according to your CUDA version (`requirements.txt` only records package names). 
`transformers` needs to support Qwen3 (>= 4.51). Resolution of 224 + batch size 32 + bf16 suggests ≥12 GB VRAM;  if insufficient, reduce `batch_size` in the config.

## 4. Quick Self-Check (Recommended to run first)

```bash
python -m clip_qwen.train --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml --debug
```

`--debug` only runs 256 training samples + 64 evaluation images to confirm that "data/weights/environment" are functioning correctly.

## 5. Training / Evaluation

```bash
# Train RSITMD (default protocol)
python -m clip_qwen.train --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml

# Train RSICD (train→test)
python -m clip_qwen.train --config clip_qwen/configs/rsicd_clip_qwen_itc.yaml

# Evaluate a specific checkpoint
python -m clip_qwen.eval --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml \
    --checkpoint outputs/<run_dir>/best.pt

# You can also run directly via the root entry point without -m
python train.py --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml
```

Outputs are saved in `outputs/<timestamp>_<config_name>/`, containing `config.yaml`, training history, best/last weights,  and round-by-round retrieval metrics (including Top-K, bidirectional) for `round_1` to `round_4` (+`round_5` negation round).

## 6. Data Format (Already Reconstructed)

Each image record contains:

```json
{
  "filename": "airport_1.jpg",           // Corresponds to the original image filename
  "imgid": 0,
  "split": "train",                      // train / val / test
  "sentences": [ {"sentid": 0, "raw": "...", "tokens": [...]}, ... ],  // Original 5 captions
  "content": ["airport terminal building", "parked airplanes", ...],   // Scene elements (phrases)
  "round1": "It looks like an airport with a long terminal building.", // Progressive round 1
  "round2": "...", "round3": "...", "round4": "...",
  "hard_negative_content": ["curved road", "parking lot", ...]         // Negative attribute pool
}
```

- Progressive protocol: Training samples by image and randomly takes `round1..roundK` prefixes; evaluation evaluates round by round,  and the gallery retains only unique images.
- The package only contains **the version of the reconstructed JSON that produced the paper results** (which is also the file directly referenced by configs), md5: 
  `datasets/RSITMD/dataset_RSITMD.json` = `d1f5a51dc6e1c089d10f28cb2ccf026d`; 
  `datasets/RSICD/dataset_rsicd.json` = `b1675d0f9313d9a9bee54daeae3ba344`.

## 7. Frequently Asked Questions

- **Error `FileNotFoundError: ... RS5M_ViT-B-32_RET-2.pt`**: Initialization weights are incomplete (see 2c).
- **Error saying images cannot be found**: Image directory name/position is incorrect, or commands were not executed in the package root directory.
- **Out of VRAM**: Reduce `batch_size` (e.g., 32→16) and `eval_batch_size`; keep `amp: true, precision: bf16`.
- **Only want to use the pure CLIP baseline**: First change `dataset_json` in its config to `datasets/RSITMD/dataset_RSITMD.json`  (the old filename `dataset_RSITMD_with_content.json` written in that config is not included in this package; 
  the new JSON already integrates the `content` field for direct compatibility), then execute  `python -m clip_rsitmd_baseline.train --config clip_rsitmd_baseline/configs/rsitmd_vitb32_sdm.yaml`  (note that it uses the SDM protocol, which is a different evaluation standard from `clip_qwen`'s progressive protocol).
