# CLIP+Qwen Negation Ablation Guide

This document describes the negation constraint ablation experiment process in the current `clip_qwen` progressive retrieval framework.

Main settings:

- No model architecture changes.
- No ITC loss changes.
- Use dataset-level `hard_negative_content` / `neg_attrs` and training/evaluation query construction to test whether negation constraints are helpful.

## 1. Current Protocol

### Training stage:

```text
random progressive round_k
```

With probability `negation_train_prob`, a random negation attribute is appended:

```text
random progressive round_k + There is no {sampled_neg_attr} in this scene.
```

### Evaluation stage:

```text
round_1 = round1
round_2 = round1 + round2
round_3 = round1 + round2 + round3
round_4 = round1 + round2 + round3 + round4
round_5 = round1 + round2 + round3 + round4 + negation
```

`round_5` uses the complete progressive query and appends a sentence generated from the top-N negation attributes:

```text
There is no attr1 in this scene.
There is no attr1 or attr2 in this scene.
There is no attr1, attr2, or attr3 in this scene.
```

## 2. Data Source

The current main pipeline directly reads the following fields from the dataset JSON:

- `hard_negative_content`
- or the compatible old field `neg_attrs`

Training and evaluation do not automatically run any offline mining scripts.

If you need to complete empty `hard_negative_content`, directly write them into the dataset JSON during data preprocessing. The current training main path no longer depends on any offline confuser mining scripts.

## 3. Configuration File

Edit:

```text
clip_qwen/configs/rsitmd_clip_qwen_itc.yaml
```

Key parameters:

```yaml
use_dataset_neg_attrs: true
negation_train_prob: 0.5
negation_eval_topn: 1
```

Meaning:

- `use_dataset_neg_attrs`: Read negation attributes from `hard_negative_content` / `neg_attrs` in the dataset JSON.
- `negation_train_prob`: Probability of adding negation constraints during training.
- `negation_eval_topn`: Number of top negation attributes used in `round_5` evaluation.

Common settings:

```yaml
negation_train_prob: 0.0  # Do not use negation during training
negation_train_prob: 0.5  # Randomly add negation for half of the samples, recommended starting point
negation_train_prob: 1.0  # Add negation for every sample with neg_attrs
```

```yaml
negation_eval_topn: 1  # round_5 uses top1 attribute, recommended for main experiments
negation_eval_topn: 2  # round_5 uses top2 attributes
negation_eval_topn: 5  # round_5 uses top5 attributes
```

## 4. Training

Training command:

```bash
python -m clip_qwen.train --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml
```

RSICD example:

```bash
python -m clip_qwen.train --config clip_qwen/configs/rsicd_clip_qwen_itc.yaml --eval-split test
```

If GPU memory is insufficient, first modify:

```yaml
batch_size: 60
max_text_length: 128
```

If OOM still occurs, reduce to:

```yaml
batch_size: 48
```

The training log will contain:

```text
avg_neg_rate=0.xx
```

It indicates the average ratio of samples with appended negation sentences during current training. If `negation_train_prob=0.5`, the value should normally be close to `0.5`.

## 5. Evaluation

After training, run evaluation:

```bash
python -m clip_qwen.eval \
  --config clip_qwen/configs/rsitmd_clip_qwen_itc.yaml \
  --checkpoint outputs/<run_dir>/best.pt
```

If test samples contain `hard_negative_content` / `neg_attrs`, evaluation will output:

```text
round_1
round_2
round_3
round_4
round_5
```

Explanation:

- `round_1` to `round_4` are normal progressive retrieval.
- `round_5` is the complete `round_4` query with additional negation constraints.

Core comparison:

```text
round_5 vs round_4
```

If `round_5 > round_4`, it indicates that additional negation constraints provide positive gains for the current model.

The baseline should also be compared:

```text
negation model round_4 vs baseline model round_4
negation model round_5 vs baseline model round_4
```

Ideal case:

```text
negation model round_4 is close to baseline round_4
negation model round_5 is higher than negation model round_4
```

This indicates that random negation training does not significantly damage normal progressive retrieval, while adding negation constraints at the final stage can bring additional improvements.

## 6. Recommended Experiment Settings

Main experiment:

```yaml
negation_train_prob: 0.5
negation_eval_topn: 1
```

Training probability ablation:

```text
negation_train_prob = 0.0 / 0.3 / 0.5 / 0.7 / 1.0
```

Evaluation top-N ablation:

```text
negation_eval_topn = 1 / 2 / 3 / 5
```

Neighbor mining number ablation:

```bash
--topk-neighbors 5
--topk-neighbors 10
--topk-neighbors 20
```

## 7. Notes

- Each time `--topk-neighbors` or `--topm-attrs` is changed, `rsitmd_clip_qwen_neg_attrs.json` needs to be regenerated.
- The old format `neg_attr` JSON cannot be used anymore.
- The improvement of `round_5` cannot be directly interpreted as the model fully understanding negation logic. It first shows that the negation constraint provides additional discriminative information for the current retrieval task.
- To strictly verify negation understanding, additional counterfactual evaluation of `There is X` vs `There is no X` is still required.