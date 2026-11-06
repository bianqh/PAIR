# CLIP RSITMD Baseline

Standardized OpenCLIP `ViT-B-32` fine-tuning baseline for RSITMD image-text retrieval, using the cleanest `sentence` text mode by default.

## Environment Dependencies

-   Python 3.10+
-   `torch`
-   `torchvision`
-   `open_clip_torch`
-   `safetensors`
-   `pyyaml`
-   `Pillow`

If these packages are not available in the environment, install them according to your current environment management method. This implementation does not depend on online model downloading.

## Data Format

Currently only supports:

-   Weights: `weights/open_clip_ViT-B-32.safetensors`
-   Dataset: `datasets/RSITMD/dataset_RSITMD_with_content.json`

The image root directory is automatically parsed to `images/` at the same level as the JSON by default. It can also be explicitly set by `image_root` in the configuration.

RSITMD loading protocol:

-   Training set uses caption-level flatten
    -   The 5 sentences of one image are expanded into 5 training samples
    -   `imgid` is used as the SDM label id
-   Test set follows the standard bidirectional retrieval protocol
    -   image gallery only keeps unique images
    -   text query uses all test captions
    -   `t2i` each caption corresponds to only 1 GT image
    -   `i2t` each image corresponds to all its GT captions

## Text Mode

Configuration item: `text_source`

-   `sentence`
    -   Only use the original caption
    -   Default value, most suitable as a clean CLIP fine-tune baseline
-   `content`
    -   Only use image-level `content` list, concatenated by commas
-   `sentence_plus_content`
    -   Use template:
        `<raw caption>. scene content: <content1>, <content2>, ...`

## Training Command

Execute in the repository root:

``` bash
python train.py --config clip_rsitmd_baseline/configs/rsitmd_vitb32_sdm.yaml
```

Train the joint retrieval model of CLIP image encoder + Qwen text encoder:

``` bash
python train.py --config clip_rsitmd_baseline/configs/rsitmd_clip_qwen_sdm.yaml
```

Quick debug:

``` bash
python train.py --config clip_rsitmd_baseline/configs/rsitmd_vitb32_sdm.yaml --debug
```

The Qwen version requires additional installation of `transformers>=4.51.0`, and `qwen_model_name_or_path` must load Qwen3 config and tokenizer files. For offline usage, place Hugging Face files such as `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt` into a local directory, then set `qwen_model_name_or_path` to this directory.

## Evaluation Command

Evaluate the best trained checkpoint:

``` bash
python eval.py --config clip_rsitmd_baseline/configs/rsitmd_vitb32_sdm.yaml --checkpoint outputs/20260321_112330_rsitmd_vitb32_sdm/best.pt
```

Only evaluate initial pretrained:

``` bash
python eval.py --config clip_rsitmd_baseline/configs/rsitmd_vitb32_sdm.yaml
```

## Output Files

The training output directory is created as `outputs/time_directory_name/`, for example `outputs/20260321_112330_rsitmd_vitb32_sdm/`, and mainly includes:

-   `config.yaml`
-   `train_YYYYMMDD_HHMMSS.log`
-   `history.json`
-   `best.pt`
-   `best_metrics.json`

The evaluation output directory is written to the corresponding run directory by default:

-   `outputs/20260321_112330_rsitmd_vitb32_sdm/eval/`

It contains:

-   `metrics.json`
-   `topk_results.json`
-   `embeddings.pt` (only when `save_embeddings: true`)
-   `eval_YYYYMMDD_HHMMSS.log`

## Metric Definition

-   `t2i`
    -   Each test caption retrieves all test images
    -   Outputs `R@1 / R@5 / R@10 / MedR / MeanR`
-   `i2t`
    -   Each test image retrieves all test captions
    -   Any GT caption hit in top-k is considered a hit
    -   Outputs `R@1 / R@5 / R@10 / MedR / MeanR`
-   `mean_recall`
    -   First calculate `mean(R@1, R@5, R@10)` for `t2i` and `i2t`
    -   Then calculate the average of the two directions

## Default Training Settings

-   backbone: `ViT-B-32`
-   loss: `sdm`
    -   Optional: `itc`
-   optimizer: `AdamW`
-   scheduler: cosine
-   `min_lr_ratio`
    -   Scaling factor of the minimum learning rate relative to `lr`
    -   For example, when `lr=1e-6` and `min_lr_ratio=0.1`, the actual
        minimum learning rate is `1e-7`
-   precision:
    -   Default `bf16`
    -   Use BF16 autocast when `CUDA + BF16 supported`
    -   Otherwise automatically fall back to FP16 autocast
    -   Fall back to FP32 on CPU
-   TF32:
    -   `allow_tf32: true`
-   `logit_scale`:
    -   Keep the parameter as default FP32
    -   Clamp after each update to `exp(logit_scale) <= 100`

## Known Notes

-   `SDM` currently follows the implementation behavior of `demo/loss.py` in the repository, making future direct comparisons easier.
-   The default checkpoint strategy only keeps the best model weight `best.pt` on the validation set. It does not save optimizer, scheduler, or scaler, and does not additionally keep `last.pt` or per-epoch checkpoint.
-   `topk_results.json` is larger than pure metric files, but it is useful for checking retrieval examples.
-   During evaluation, the final embedding is converted to FP32 before calculating retrieval metrics, making the results more stable.
