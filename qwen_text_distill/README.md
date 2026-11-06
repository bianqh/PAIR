# Qwen Text Distillation

This folder is for distilling the RS5M/GeoRSCLIP text embedding space into the
Qwen text tower used by the retrieval model.

## Goal

Teacher:

```text
RS5M CLIP text encoder -> normalized 512-d text feature
```

Student:

```text
Qwen text encoder -> mean/last pooling -> text projection -> normalized 512-d text feature
```

Loss:

```text
1 - cosine(student, teacher)
```

The output checkpoint only needs the student side weights:

```text
qwen
text_projection
```

## Data

Put RS5M metadata CSV files under:

```text
datasets/RS5M_meta/
```

Recommended first files:

```text
pub11_train_metadata.csv
pub11_validation_metadata.csv
rs3_train_val_metadata.csv
```

The downloader paths are:

```text
https://huggingface.co/datasets/Zilun/RS5M/resolve/main/pub11_train_metadata.csv
https://huggingface.co/datasets/Zilun/RS5M/resolve/main/pub11_validation_metadata.csv
https://huggingface.co/datasets/Zilun/RS5M/resolve/main/rs3_train_val_metadata.csv
```

## Run

After the CSV files are downloaded:

```bash
python qwen_text_distill/train_distill.py --config qwen_text_distill/configs/rs5m_qwen_text_distill.yaml
```

For a smoke test:

```bash
python qwen_text_distill/train_distill.py --config qwen_text_distill/configs/rs5m_qwen_text_distill.yaml --debug
```
