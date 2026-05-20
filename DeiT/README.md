# Supervised Vision Transformers with VIOLIN-DeiT models.

For details on the original DeiT models, please refer to the following papers:
<details>
<summary>
  <a href="https://arxiv.org/pdf/2012.12877">DeiT:</a> Data-Efficient Image Transformers, ICML 2021 [<b>bib</b>]
</summary>

```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
</details>


## Results

### Pretraining on ImageNet
| **Model**      | Params | Top-1 Accuracy (%) | Δ    |
| -------------- | ------ | ------------------ | ---- |
| **DeiT-T**     | 5M     | 72.2 → **73.0**    | +0.8 |
| **DeiT-S**     | 22M    | 79.8 → **80.7**    | +0.9 |
| **DeiT-B**     | 86M    | 81.8 → **81.9**    | +0.1 |

### Fine-tuning on VTAB-1K
| **Model**      | Method                        | Natural           | Specialized       | Structured        | Avg.              | Δ Avg. |
| -------------- | ----------------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------ |
| **DeiT-T**     | Baseline                      | 75.18             | 73.04             | 53.57             | 65.52             | —      |
|                | Baseline ⊙ M<sub>VIOLIN</sub> | 75.18 → **77.68** | 73.04 → **74.44** | 53.57 → **57.50** | 65.52 → **68.33** | +2.81  |
|                | Pretrained VIOLIN             | 75.18 → **76.60** | 73.04 → **73.19** | 53.57 → **54.52** | 65.52 → **66.41** | +0.89  |
|                |                               |                   |                   |                   |                   |        |
| **DeiT-S**     | Baseline                      | 78.89             | 75.88             | 53.44             | 67.38             | —      |
|                | Baseline ⊙ M<sub>VIOLIN</sub> | 78.89 → **81.50** | 75.88 → **76.73** | 53.44 → **58.26** | 67.38 → **70.46** | +3.08  |
|                | Pretrained VIOLIN             | 78.89 → **80.80** | 75.88 → **76.28** | 53.44 → **56.31** | 67.38 → **69.30** | +1.92  |
|                |                               |                   |                   |                   |                   |        |
| **DeiT-B**     | Baseline                      | 82.15             | 77.53             | 57.00             | 70.35             | —      |
|                | Baseline ⊙ M<sub>VIOLIN</sub> | 82.15 → **83.36** | 77.53 → **78.14** | 57.00 → **61.89** | 70.35 → **72.95** | +2.60  |
|                | Pretrained VIOLIN             | 82.15 → **81.93** | 77.53 → **77.18** | 57.00 → **58.90** | 70.35 → **70.99** | +0.64  |
|                |                               |                   |                   |                   |                   |        |
| **DeiT-III-S** | Baseline                      | 80.39             | 75.62             | 52.92             | 67.57             | —      |
|                | Baseline ⊙ M<sub>VIOLIN</sub> | 80.39 → **82.60** | 75.62 → **77.09** | 52.92 → **61.61** | 67.57 → **72.31** | +4.74  |
|                |                               |                   |                   |                   |                   |        |
| **DeiT-III-B** | Baseline                      | 83.34             | 77.66             | 56.71             | 70.63             | —      |
|                | Baseline ⊙ M<sub>VIOLIN</sub> | 83.34 → **84.75** | 77.66 → **78.41** | 56.71 → **63.03** | 70.63 → **73.94** | +3.31  |


## Training

While `curves.py` file includes the definitions of each curve, `models_violin.py` creates the VIOLIN-DeiT models. Pretrained models will be released soon.

### VIOLIN-DeiT training 

VIOLIN-DeiT-tiny:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model violin_tiny_patch16 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

VIOLIN-DeiT-small:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model violin_small_patch16 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

#### Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train the VIOLIN-DeiT-base model on ImageNet on 2 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model violin_base_patch16 --data-path /path/to/imagenet
```

Training logs are located in [this folder]((../logs)).

## ⚙️ Ablation Studies

We provide several configurable options to adapt **VIOLIN** to your training setup. These options can be passed as command-line arguments when running scripts for DeiT.

### Available Arguments

| Argument            | Default         | Description |
|---------------------|-----------------|-------------|
| `--cls_tok`         | `False`         | Use a `[CLS]` token for classification instead of GAP. |
| `--pos_emb`         | `True`          | Use the default positional embeddings. |
| `--curves`          | All 8 curves    | List of curves to use. Example: `snake,snake_T,hilbert`. |
| `--scale`           | `True`          | Learn a scalar weight for the attention mask. |
| `--initialize`      |`True`           | Initialize the mask values close to 1. |
| `--mask`            |`learned`        | Use either learned or fixed mask. Options: `learned` or `fixed`.  |
| `--method`          |`mul_v1`         | Method to apply the mask to attention.   Options: `mul_v1`, `mul_v2`, `add_v1`, `mul_after_sm`, or `add_after_sm`.|


#### Default Curve Set

Unless overridden, VIOLIN uses the following 8 space-filling curves: `[snake, snake_T, zigzag, zigzag_T, peano, peano_T, hilbert, hilbert_T]`

#### Descriptions of Methods
| Strategy                                 | Description                                      |
|------------------------------------------|--------------------------------------------------|
| `mul_v1`      | $S(\mathbf{A} \odot \mathbf{M})$ |
| `mul_v2`      |$S(\mathbf{A}' \odot (\mathbf{I} + \mathbf{M}))$         |
| `add_v1`      | $S(\mathbf{M} + \mathbf{A}')$                  |
| `mul_after_sm`      | $S(\mathbf{A}') \odot \mathbf{M}$ |
| `add_after_sm`          | $S(\mathbf{A}') + \mathbf{M}$            |

