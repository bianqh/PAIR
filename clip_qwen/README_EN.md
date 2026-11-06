# CLIP+Qwen Progressive Retrieval

CLIP image encoder + Qwen text encoder progressive query retrieval
training code.

## Environment Dependencies

-   Python 3.10+
-   `torch`
-   `torchvision`
-   `open_clip_torch`
-   `transformers`
-   `safetensors`
-   `pyyaml`
-   `Pillow`

If these packages are not available in the environment, install them according to your current environment management method. This implementation does not depend on online model downloading.

## Data Format

The current main pipeline supports:

-   CLIP weights: `weights/RS5M_ViT-B-32_RET-2.pt`
-   Qwen student weights: `weights/best_student.pt`
-   Dataset: `datasets/RSITMD/dataset_RSITMD.json`
-   Dataset: `datasets/RSICD/dataset_rsicd.json`

The image root directory is automatically parsed to `images/` at the same level as the JSON by default. It can also be explicitly set by `image_root` in the configuration.

Dataset loading protocol:

-   In progressive mode, training samples are constructed at the image level.
    -   Each image uses progressive descriptions from `round1` to `round4`.
    -   Progressive round prefixes are randomly sampled during training.
-   The evaluation set is evaluated by progressive rounds.
    -   The image gallery only keeps unique images.
    -   Each image generates one text query for each round.
    -   When `eval_progressive_rounds: 4`, outputs are `round_1` to `round_4`.

## Text Mode

Progressive training is controlled by `use_progressive_dialogue: true`. After enabling it, training and evaluation texts come from `round1` to `round4`, and the original 5-caption flatten protocol is no longer used.

Progressive answers are encoded by default into a fixed natural question + answer format:

``` text
Briefly describe the overall scene you are looking for? ...
What are the main objects or regions in the scene? ...
How are these objects spatially arranged? ...
Are there any additional details that help identify the scene? ...
```

## Training Command

Execute in the repository root:

``` bash
python -m clip_qwen.train --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml
```

Quick debug:

``` bash
python -m clip_qwen.train --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml --debug
```

RSICD training example:

``` bash
python -m clip_qwen.train --config clip_qwen/configs/rsicd_clip_qwen_itc.yaml --eval-split test
```

## Negation ablation

The current negation method only uses `hard_negative_content` / `neg_attrs` inside the dataset. During training, each sampled progressive query randomly samples negative attributes from the corresponding sample according to `negation_train_prob` and appends:

``` text
There is no {sampled_neg_attr} in this scene.
```

Progressive evaluation first evaluates `round_1` to `round_4` following the normal protocol, and then additionally evaluates `round_5`. `round_5` uses the complete progressive query and combines the first N negative attributes according to `negation_eval_topn` into one negation constraint sentence, such as `There is no attr1 or attr2 in this scene.`

`round_5` additionally encodes the fixed exclusion question:

``` text
What is absent from your target scene that may appear in similar scenes? ...
```

The Qwen version requires additional installation of `transformers>=4.51.0`, and `qwen_model_name_or_path` must be able to load the Qwen3 config and tokenizer files. If offline usage is required, it is recommended to place `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, and other files from Hugging Face into a local directory, then set `qwen_model_name_or_path` to this directory.

## Evaluation Command

Evaluate the best trained checkpoint:

``` bash
python -m clip_qwen.eval --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml --checkpoint outputs/<run_dir>/best.pt
```

## Output Files

The training output directory is created as `outputs/dataset_name/time_directory_name/`, for example `outputs/RSITMD/20260419_220000_rsitmd_clip_qwen_progressive/`, and mainly includes:

-   `config.yaml`
-   `train_YYYYMMDD_HHMMSS.log`
-   `history.json`
-   `best.pt`
-   `best_metrics.json`

## Metric Definition

-   `t2i`
    -   Each progressive query retrieves all test images.
    -   Outputs `R@1 / R@5 / R@10 / MedR / MeanR`
-   `i2t`
    -   Each test image retrieves all progressive queries.
    -   Outputs `R@1 / R@5 / R@10 / MedR / MeanR`
-   `mean_recall`
    -   First calculate `mean(R@1, R@5, R@10)` for `t2i` and `i2t`.
    -   Then calculate the average of the two directions.

## Default Training Settings

-   backbone: `ViT-B-32`
-   loss: `itc`
-   optimizer: `AdamW`
-   scheduler: cosine
-   `min_lr_ratio`
    -   Scaling factor of the minimum learning rate relative to `lr`.
    -   For example, when `lr=1e-6` and `min_lr_ratio=0.1`, the actual
        minimum learning rate is `1e-7`.
-   precision:
    -   Default `bf16`
    -   Use BF16 autocast when `CUDA + BF16 supported`.
    -   Otherwise automatically fall back to FP16 autocast.
    -   Fall back to FP32 on CPU.
-   TF32:
    -   `allow_tf32: true`
-   `logit_scale`:
    -   Keep the parameter in default FP32.
    -   Clamp after each update so that
        `exp(logit_scale) <= logit_scale_max`.

## Known Notes

-   The default checkpoint strategy only keeps the best model weight `best.pt` on the evaluation set. It does not save optimizer, scheduler, or scaler, and does not keep additional `last.pt` or per-epoch checkpoints.
-   `topk_results.json` is larger than pure metric files, but it is useful for checking retrieval examples.
-   During evaluation, the final embedding is converted to FP32 before calculating retrieval metrics, making the results more stable.
