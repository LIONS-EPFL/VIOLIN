# Self-Supervised Vision Transformers with VIOLIN-DINO

For details on original DINO models, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

## Results

### Pretraining on ImageNet
| **Model**  | Epochs | KNN Accuracy (%) | Δ KNN | Linear Accuracy (%) | Δ Linear |
| ---------- | ------ | ---------------- | ----- | ------------------- | -------- |
| **DINO-S** | 100    | 69.3 → **70.0**  | +0.7  | 74.0 → **74.6**     | +0.6     |
|            | 300    | 72.8 → **73.4**  | +0.6  | 76.1 → **76.4**     | +0.3     |
| **DINO-B** | 300    | 76.1 → **76.1**  | —     | 78.2 → **78.4**     | +0.2     |

### Fine-tuning results on VTAB-1K
| **Model**  | Method                        | Natural           | Specialized       | Structured        | Avg.              | Δ Avg. |
| ---------- | ----------------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------ |
| **DINO-S** | Baseline                      | 74.00             | 66.48             | 50.89             | 62.39             | —      |
|            | Baseline ⊙ M<sub>VIOLIN</sub> | 74.00 → **75.91** | 66.48 → **66.74** | 50.89 → **52.65** | 62.39 → **63.70** | +1.31  |
|            | Pretrained VIOLIN             | 74.00 → **75.18** | 66.48 → **66.84** | 50.89 → **51.15** | 62.39 → **62.87** | +0.48  |
|            |                               |                   |                   |                   |                   |        |
| **DINO-B** | Baseline                      | 76.12             | 72.86             | 47.73             | 63.31             | —      |
|            | Baseline ⊙ M<sub>VIOLIN</sub> | 76.12 → **77.63** | 72.86 → **73.22** | 47.73 → **49.69** | 63.31 → **64.70** | +1.39  |
|            | Pretrained VIOLIN             | 76.12 → **77.48** | 72.86 → **73.22** | 47.73 → **50.81** | 63.31 → **65.13** | +1.82  |

## Training

While `curves.py` file includes the definitions of each curve, `violin_models.py` creates the VIOLIN-DINO models. Pretrained models will be released soon.

### VIOLIN-DINO training 

```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch violin_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

### Multi-node training
We use Slurm and [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`). To train on 2 nodes with 8 GPUs each (total 16 GPUs):

DINO with VIOLIN-small network.

```
python run_with_submitit.py --arch violin_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

DINO with VIOLIN-base network.

```
python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch violin_base  --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

Training logs for small and base models are located in [this folder]((../logs/DINO)).

## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --arch violin_small --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet 
```
k-NN evaluation results for small and base models are located in [this folder]((../logs/DINO)).

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights of the small violin model on a single node with 8 GPUs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --arch violin_small --pretrained_weights /path/to/checkpoint --data_path /path/to/imagenet
```
For the base model, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --arch violin_base --lr 5e-4 --n_last_blocks 1 --pretrained_weights /path/to/checkpoint.pth --data_path /path/to/imagenet
```
Linear classifier evaluation logs for small and base models are located in [this folder]((../logs/DINO)).

## Self-attention video generation

<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 16x16 patches trained with DINO" src="../assets/cats.gif">
  <p><strong>Video:</strong> Self-attention video generation with VIOLIN models.</p>
</div>

You can generate videos with VIOLIN models too, using `video_generation_violin.py`.

Extract frames from input video and generate attention video:
```
python video_generation.py  --arch violin_small --pretrained_weights /path/to/checkpoint.pth \
    --input_path input/video.mp4 \
    --output_path output/ \
    --fps 25
```

Use folder of frames already extracted and generate attention video:
```
python video_generation.py  --arch violin_small --pretrained_weights /path/to/checkpoint.pth \
    --input_path output/frames/ \
    --output_path output/ \
    --resize 256 \
```

Only generate video from folder of attention maps images:
```
python video_generation.py --arch violin_small --input_path output/attention \
    --output_path output/ \
    --video_only \
    --video_format avi
```

## ⚙️ Ablation Studies

We provide several configurable options to adapt **VIOLIN** to your training setup. These options can be passed as command-line arguments when running scripts for DINO in all scales.

### Available Arguments

| Argument            | Default         | Description |
|---------------------|-----------------|-------------|
| `--cls_tok`         | `True`         | Use a `[CLS]` token for classification instead of GAP. |
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


## Citation of the original DINO
```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
